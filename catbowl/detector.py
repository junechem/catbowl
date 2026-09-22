"""Deciding whether *a cat* is at the bowl, and where in the frame it is.

Four strategies:

``motion``
    Background subtraction. Cheap (a couple of milliseconds), and with a fixed
    camera it is a good proxy for "something just walked up". It has no idea
    what a cat is: a hand, a dog or a swaying curtain all pass.

``ssdlite``
    A real COCO object detector, filtered to the ``cat`` class. Much more
    selective - it will not fire on a moving curtain - but costs a few hundred
    milliseconds per frame on a Pi 4.

``hybrid``
    Motion as a cheap trigger, ssdlite as the gate. This is the default, and the
    reason is worth stating plainly: the classifier is a logistic regression
    over your cats, so its probabilities always sum to one and it *must* return
    one of them for whatever it is shown. Handed a dog, a hand or a carrier bag
    it will answer "pepper", sometimes with high confidence - out-of-distribution
    inputs are exactly where these models are confidently wrong, so the
    confidence floor does not save you. Something upstream has to establish that
    the thing at the bowl is a cat at all, and only then ask which cat.

    Running ssdlite on every frame does that but costs a few hundred
    milliseconds per frame on a Pi 4, so it runs only when motion first
    appears. A "not a cat" answer suppresses it for ``reject_backoff_s`` so a
    swaying curtain cannot pin the CPU at full rate.

    Once it says yes, that starts a *visit*, and a visit ends only when it is
    actively disproven - not when a timer runs out. This matters more than it
    sounds: a cat with its head down in a bowl does not look like a cat to a
    COCO detector, and it will keep not looking like one for as long as it
    keeps eating. A gate that demanded periodic re-proof would revoke the
    animal mid-meal and drop the lid on it. So ssdlite is re-asked every
    ``confirm_every_s`` to catch a swap, but only an unbroken
    ``confirm_grace_s`` of refusals ends the visit, and motion must be gone for
    ``visit_gap_s`` before the visit is considered over.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .config import DetectorConfig

log = logging.getLogger(__name__)

COCO_CAT_ID = 17


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # x, y, w, h in frame pixels
    score: float
    source: str
    # How many cats the frame held, when something counted them. A detector
    # that cannot count (motion, none) leaves this 1, and callers must not
    # read that as "exactly one cat" - only as "no second cat was reported".
    crowd: int = 1

    def crop(self, image: np.ndarray, pad_frac: float = 0.0) -> np.ndarray:
        x, y, w, h = self.bbox
        if pad_frac:
            px, py = int(w * pad_frac), int(h * pad_frac)
            x, y, w, h = x - px, y - py, w + 2 * px, h + 2 * py
        H, W = image.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return image
        return np.ascontiguousarray(image[y0:y1, x0:x1])


def _union(a: tuple[int, int, int, int],
           b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The smallest x, y, w, h box containing both."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x, y = min(ax, bx), min(ay, by)
    return x, y, max(ax + aw, bx + bw) - x, max(ay + ah, by + bh) - y


class Detector:
    def detect(self, image: np.ndarray) -> Detection | None:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        """Forget accumulated scene state (called after the lid closes)."""


class NullDetector(Detector):
    """Always reports the whole frame. Use when the camera only sees the bowl."""

    def detect(self, image: np.ndarray) -> Detection | None:
        h, w = image.shape[:2]
        return Detection((0, 0, w, h), 1.0, "none")


class MotionDetector(Detector):
    def __init__(self, cfg: DetectorConfig):
        import cv2

        self._cv2 = cv2
        self.cfg = cfg
        self._make_subtractor()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._seen = 0

    def _make_subtractor(self) -> None:
        self._bg = self._cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )
        self._seen = 0

    def detect(self, image: np.ndarray) -> Detection | None:
        cv2 = self._cv2
        small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        mask = self._bg.apply(small)
        self._seen += 1
        if self._seen <= self.cfg.warmup_frames:
            return None   # still learning what "empty" looks like

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        frame_area = small.shape[0] * small.shape[1]
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        if area < self.cfg.min_area_frac * frame_area:
            return None

        x, y, w, h = cv2.boundingRect(biggest)
        return Detection((x * 2, y * 2, w * 2, h * 2), float(area / frame_area), "motion")

    def reset(self) -> None:
        self._make_subtractor()


class SsdliteCatDetector(Detector):
    """torchvision's ssdlite320_mobilenet_v3_large, filtered to cats."""

    def __init__(self, cfg: DetectorConfig):
        import torch
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        self._torch = torch
        self.cfg = cfg
        torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.eval()

    def detect(self, image: np.ndarray) -> Detection | None:
        torch = self._torch
        rgb = image[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        with torch.inference_mode():
            out = self.model([tensor])[0]

        best: Detection | None = None
        crowd = 0
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            if int(label) != COCO_CAT_ID or float(score) < self.cfg.score_threshold:
                continue
            crowd += 1
            x0, y0, x1, y1 = (int(v) for v in box.tolist())
            det = Detection((x0, y0, x1 - x0, y1 - y0), float(score), "ssdlite")
            if best is None or det.score > best.score:
                best = det
        if best is None:
            return None
        # The crop stays the best single cat: the classifier is asked "which cat
        # is this one", and the count travels separately for the caller to veto
        # on. Merging two cats into one box would just make an unreadable crop.
        return replace(best, crowd=crowd)


class YoloCatDetector(Detector):
    """YOLO11n, exported to ONNX and run by OpenCV's dnn module, filtered to cats.

    Twice ssdlite's cost and far better at the job: COCO mAP 39 against 21, and
    it is the dark, low-contrast cat that the small detector loses first. The
    ONNX file is exported once on a workstation (`yolo export model=yolo11n.pt
    format=onnx imgsz=320`) so the Pi needs neither ultralytics nor torch for it.
    """

    CAT = 15                  # "cat" among YOLO's 80 COCO classes
    NMS_IOU = 0.45

    def __init__(self, cfg: DetectorConfig):
        import cv2

        self._cv2 = cv2
        self.cfg = cfg
        self.net = cv2.dnn.readNetFromONNX(cfg.yolo_path)
        # The input side is fixed at export time; read it off the model's name
        # rather than trusting a second setting to agree with it.
        digits = "".join(c for c in cfg.yolo_path.rsplit("-", 1)[-1] if c.isdigit())
        self.size = int(digits) if digits else 320

    def detect(self, image: np.ndarray) -> Detection | None:
        cv2 = self._cv2
        H, W = image.shape[:2]
        # Letterbox: scale the long side to the input and pad the rest, so the
        # cat is not squashed out of the shape YOLO was trained on.
        scale = self.size / max(H, W)
        w, h = round(W * scale), round(H * scale)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - h) // 2, (self.size - w) // 2
        canvas[top:top + h, left:left + w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()[0]           # (84, N): cx, cy, w, h, then 80 class scores

        scores = out[4 + self.CAT]
        keep = scores >= self.cfg.score_threshold
        if not keep.any():
            return None
        cx, cy, bw, bh = out[:4, keep]
        scores = scores[keep]
        x0 = (cx - bw / 2 - left) / scale
        y0 = (cy - bh / 2 - top) / scale
        boxes = np.stack([x0, y0, bw / scale, bh / scale], axis=1)
        picked = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                  self.cfg.score_threshold, self.NMS_IOU)
        picked = np.asarray(picked).reshape(-1)
        if not len(picked):
            return None
        best = int(picked[np.argmax(scores[picked])])
        x, y, bw_, bh_ = boxes[best]
        x0_, y0_ = max(0, int(x)), max(0, int(y))
        x1_, y1_ = min(W, int(x + bw_)), min(H, int(y + bh_))
        return Detection((x0_, y0_, x1_ - x0_, y1_ - y0_), float(scores[best]), "yolo",
                         crowd=len(picked))


def build_cat_detector(cfg: DetectorConfig) -> Detector:
    """The object detector named by ``cfg.model``: ssdlite or yolo."""
    if cfg.model == "yolo":
        return YoloCatDetector(cfg)
    return SsdliteCatDetector(cfg)


class HybridCatDetector(Detector):
    """Motion triggers; ssdlite decides whether it was a cat.

    The two boxes are not interchangeable and the difference matters more than
    it looks. ssdlite's is the whole animal; the motion box is only the part of
    it that moved since the last frame, which for a cat settled at a bowl is a
    twitching tail or one ear. Cropping to that gives the classifier a tuft of
    fur and asks it which cat that is.

    So the last confirmed cat box is remembered for the length of a visit, and
    every frame in between is reported as that box united with the current
    motion - never smaller than the animal ssdlite actually saw, and still
    following it if it shifts.
    """

    def __init__(
        self,
        cfg: DetectorConfig,
        motion: Detector | None = None,
        confirm: Detector | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self._motion = motion if motion is not None else MotionDetector(cfg)
        # Built lazily: loading ssdlite costs seconds and a few hundred MB, and
        # a rig running --no-model on a spare Pi should not pay for it twice.
        self._confirm = confirm
        self._clock = clock
        self._visiting = False
        self._last_motion_at = 0.0
        self._last_yes_at = 0.0
        self._next_check_at = 0.0
        self._rejected_until = 0.0
        # The most recent box ssdlite drew round the animal, for as long as the
        # visit lasts. Refreshed every confirm_every_s, and dropped when the cat
        # leaves so the next visit cannot inherit the last cat's outline.
        self._last_cat: Detection | None = None

    def _detector(self) -> Detector:
        if self._confirm is None:
            self._confirm = build_cat_detector(self.cfg)
        return self._confirm

    def detect(self, image: np.ndarray) -> Detection | None:
        now = self._clock()
        motion = self._motion.detect(image)

        if motion is None:
            # Do not end the visit on the first still frame. A cat that has
            # settled down to eat moves very little, and background
            # subtraction stops reporting it long before it has left.
            if self._visiting and now - self._last_motion_at >= self.cfg.visit_gap_s:
                self._visiting = False
                self._last_cat = None
            return None

        self._last_motion_at = now

        if not self._visiting:
            if now < self._rejected_until:
                return None            # recently judged not-a-cat; stay cheap
            cat = self._detector().detect(image)
            if cat is None:
                self._rejected_until = now + self.cfg.reject_backoff_s
                return None
            self._visiting = True
            self._last_yes_at = now
            self._rejected_until = 0.0
            self._next_check_at = now + self.cfg.confirm_every_s
            self._last_cat = cat
            return cat

        # In a visit. Re-ask periodically so a cat swapped for a dog is caught,
        # but treat a refusal as weak evidence: it is the normal answer for a
        # head-down cat. Only a sustained run of them ends the visit.
        if now < self._next_check_at:
            return self._whole_animal(motion)
        self._next_check_at = now + self.cfg.confirm_every_s
        cat = self._detector().detect(image)
        if cat is not None:
            self._last_yes_at = now
            self._last_cat = cat
            return cat
        if now - self._last_yes_at >= self.cfg.confirm_grace_s:
            self._visiting = False
            self._last_cat = None
            self._rejected_until = now + self.cfg.reject_backoff_s
            return None
        # ssdlite refused, which is the ordinary answer for a head-down cat. The
        # animal is still there, so the box it was last seen with still applies.
        return self._whole_animal(motion)

    def _whole_animal(self, motion: Detection) -> Detection:
        """*motion*, widened to include the whole cat ssdlite last drew.

        Union rather than replacement: the remembered box says how big the
        animal is, the motion box says where it is now, and a cat that shifts
        along the bowl between confirmations needs both. Only ever the latest
        confirmed box, so this cannot creep outwards over a long visit.
        """
        if self._last_cat is None:
            return motion
        x, y, w, h = _union(motion.bbox, self._last_cat.bbox)
        # The head count comes from the last confirmation too: motion cannot
        # count cats, so between checks the answer is the last real one.
        return Detection((x, y, w, h), motion.score, "hybrid", self._last_cat.crowd)

    def reset(self) -> None:
        self._motion.reset()
        self._last_cat = None
        self._visiting = False
        self._last_motion_at = 0.0
        self._last_yes_at = 0.0
        self._next_check_at = 0.0
        self._rejected_until = 0.0


def build_detector(cfg: DetectorConfig) -> Detector:
    if cfg.type == "none":
        return NullDetector()
    if cfg.type == "ssdlite":
        return build_cat_detector(cfg)
    if cfg.type == "hybrid":
        return HybridCatDetector(cfg)
    return MotionDetector(cfg)
