"""Deciding whether *something* is at the bowl, and where in the frame it is.

Two strategies:

``motion``
    Background subtraction. Cheap (a couple of milliseconds), and with a fixed
    camera it is a good proxy for "a cat just walked up". This is the default on
    a Pi 4 because it leaves the whole CPU budget for recognition.

``ssdlite``
    A real COCO object detector, filtered to the ``cat`` class. Much more
    selective - it will not fire on a moving curtain - but costs a few hundred
    milliseconds per frame on a Pi 4. Worth it if motion gives false triggers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig

log = logging.getLogger(__name__)

COCO_CAT_ID = 17


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # x, y, w, h in frame pixels
    score: float
    source: str

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
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            if int(label) != COCO_CAT_ID or float(score) < self.cfg.score_threshold:
                continue
            x0, y0, x1, y1 = (int(v) for v in box.tolist())
            det = Detection((x0, y0, x1 - x0, y1 - y0), float(score), "ssdlite")
            if best is None or det.score > best.score:
                best = det
        return best


def build_detector(cfg: DetectorConfig) -> Detector:
    if cfg.type == "none":
        return NullDetector()
    if cfg.type == "ssdlite":
        return SsdliteCatDetector(cfg)
    return MotionDetector(cfg)
