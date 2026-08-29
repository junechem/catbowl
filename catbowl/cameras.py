"""Camera plumbing.

Two bowls may point at the same physical camera (the single wide-angle layout)
or each have their own (the three-webcam layout). Either way a device is opened
exactly once by the :class:`CameraHub`; a grabber thread keeps only the newest
frame so a slow inference pass never works its way through a stale backlog.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


@dataclass
class Frame:
    image: np.ndarray   # BGR, already rotated/flipped/cropped for this bowl
    timestamp: float
    index: int


# --------------------------------------------------------------------------- #
# Capture backends
# --------------------------------------------------------------------------- #

class _Capture:
    """Minimal capture interface: read() -> BGR frame or None."""

    def read(self) -> np.ndarray | None:  # pragma: no cover - interface
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover - interface
        pass


class OpenCVCapture(_Capture):
    """USB webcam (or any V4L2 device) via OpenCV."""

    def __init__(self, device: Any, width: int, height: int, fps: int):
        import cv2

        self._cv2 = cv2
        # MJPEG keeps three cameras inside the Pi 4's shared USB 2.0 budget;
        # raw YUYV at 640x480x10fps from three cameras will not fit.
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2 if isinstance(device, int) else cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise CameraError(f"could not open camera {device!r}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class PiCameraCapture(_Capture):
    """CSI ribbon camera via picamera2 (Raspberry Pi OS Bookworm and later)."""

    def __init__(self, index: int, width: int, height: int, fps: int):
        from picamera2 import Picamera2

        self.cam = Picamera2(camera_num=index)
        cfg = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": float(fps)},
        )
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.0)  # let auto-exposure and white balance settle

    def read(self) -> np.ndarray | None:
        return self.cam.capture_array()   # picamera2's RGB888 is BGR in memory

    def release(self) -> None:
        self.cam.stop()
        self.cam.close()


class FileCapture(_Capture):
    """A video file or a directory of images - handy for replaying a recording."""

    def __init__(self, path: str, loop: bool = True, fps: int = 10):
        import cv2

        self._cv2 = cv2
        self.loop = loop
        self.delay = 1.0 / max(fps, 1)
        self._last = 0.0
        target = Path(path)
        if target.is_dir():
            self.images = sorted(
                p for p in target.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not self.images:
                raise CameraError(f"no images found in {target}")
            self.cap = None
            self.pos = 0
        else:
            self.images = []
            self.cap = cv2.VideoCapture(str(target))
            if not self.cap.isOpened():
                raise CameraError(f"could not open video {target}")

    def read(self) -> np.ndarray | None:
        # Pace playback so a recording behaves like a live camera.
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

        if self.cap is None:
            if self.pos >= len(self.images):
                if not self.loop:
                    return None
                self.pos = 0
            frame = self._cv2.imread(str(self.images[self.pos]))
            self.pos += 1
            return frame
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class SyntheticCapture(_Capture):
    """Moving blob on a static background. Used by the self-test and by tests."""

    PALETTE = [(210, 200, 190), (60, 90, 200), (70, 190, 90), (200, 90, 200)]

    def __init__(self, width: int = 640, height: int = 480, fps: int = 10, variant: int = 0):
        self.width, self.height = width, height
        self.delay = 1.0 / max(fps, 1)
        self.n = 0
        self.color = self.PALETTE[variant % len(self.PALETTE)]
        rng = np.random.default_rng(0)
        self.background = (rng.integers(40, 70, (height, width, 3))).astype(np.uint8)

    PHASE_FRAMES = 30

    def read(self) -> np.ndarray | None:
        time.sleep(self.delay)
        frame = self.background.copy()
        # The blob drifts across the frame while it is present, then leaves for a
        # while. Standing perfectly still would be absorbed into the background
        # model - which is exactly what a real cat does not do.
        step = self.n % self.PHASE_FRAMES
        present = (self.n // self.PHASE_FRAMES) % 2 == 0
        if present:
            cx = int(self.width * (0.32 + 0.36 * step / self.PHASE_FRAMES))
            cy = int(self.height * (0.55 + 0.06 * np.sin(step * 0.6)))
            r = int(min(self.width, self.height) * 0.2)
            y, x = np.ogrid[: self.height, : self.width]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
            frame[mask] = self.color
        self.n += 1
        return frame


def open_capture(cfg: CameraConfig) -> _Capture:
    """Build the capture backend named by ``cfg.device``."""
    device = cfg.device
    if isinstance(device, str):
        if device.startswith("synthetic"):
            # "synthetic" or "synthetic:2" - the suffix picks a distinct blob colour
            _, _, variant = device.partition(":")
            return SyntheticCapture(cfg.width, cfg.height, cfg.fps, int(variant or 0))
        if device.startswith("csi:"):
            return PiCameraCapture(int(device.split(":", 1)[1]), cfg.width, cfg.height, cfg.fps)
        if device.startswith("file:"):
            return FileCapture(device.split(":", 1)[1], fps=cfg.fps)
        if device.isdigit():
            device = int(device)
    return OpenCVCapture(device, cfg.width, cfg.height, cfg.fps)


# --------------------------------------------------------------------------- #
# Hub
# --------------------------------------------------------------------------- #

class _Stream:
    """One device, one grabber thread, one always-current frame."""

    def __init__(self, cfg: CameraConfig, capture_factory=open_capture):
        self.cfg = cfg
        self.capture = capture_factory(cfg)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._index = 0
        self._failures = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{cfg.key}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.capture.read()
            except Exception:
                log.exception("camera %s raised while reading", self.cfg.key)
                frame = None
            if frame is None:
                self._failures += 1
                if self._failures in (1, 10, 100) or self._failures % 500 == 0:
                    log.warning("camera %s returned no frame (%d in a row)",
                                self.cfg.key, self._failures)
                time.sleep(0.1)
                continue
            self._failures = 0
            with self._lock:
                self._frame = frame
                self._ts = time.monotonic()
                self._index += 1

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        with self._lock:
            if self._frame is None:
                return None, 0.0, 0
            return self._frame, self._ts, self._index

    @property
    def healthy(self) -> bool:
        return self._failures < 50

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.capture.release()
        except Exception:  # pragma: no cover - best effort on shutdown
            log.exception("error releasing camera %s", self.cfg.key)


class CameraHub:
    """Opens each distinct device once and hands out per-bowl views of it."""

    def __init__(self, capture_factory=open_capture):
        self._capture_factory = capture_factory
        self._streams: dict[str, _Stream] = {}
        self._lock = threading.Lock()

    def view(self, cfg: CameraConfig) -> "CameraView":
        with self._lock:
            stream = self._streams.get(cfg.key)
            if stream is None:
                log.info("opening camera %s", cfg.key)
                stream = _Stream(cfg, self._capture_factory)
                self._streams[cfg.key] = stream
        return CameraView(stream, cfg)

    def healthy(self) -> dict[str, bool]:
        return {key: stream.healthy for key, stream in self._streams.items()}

    def close(self) -> None:
        with self._lock:
            for stream in self._streams.values():
                stream.close()
            self._streams.clear()


class CameraView:
    """A bowl's window onto a shared stream: rotation, flip and ROI crop."""

    def __init__(self, stream: _Stream, cfg: CameraConfig):
        self._stream = stream
        self.cfg = cfg
        self._last_index = -1

    def read(self, only_new: bool = False) -> Frame | None:
        image, ts, index = self._stream.latest()
        if image is None:
            return None
        if only_new and index == self._last_index:
            return None
        self._last_index = index
        return Frame(image=transform(image, self.cfg), timestamp=ts, index=index)

    def wait_for_frame(self, timeout: float = 5.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.05)
        return None


def transform(image: np.ndarray, cfg: CameraConfig) -> np.ndarray:
    """Apply rotation, mirroring and the ROI crop, in that order."""
    import cv2

    if cfg.rotate == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif cfg.rotate == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif cfg.rotate == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if cfg.flip:
        image = cv2.flip(image, 1)
    if cfg.roi:
        h, w = image.shape[:2]
        x, y, rw, rh = cfg.roi
        x0, y0 = int(x * w), int(y * h)
        x1, y1 = min(w, x0 + max(1, int(rw * w))), min(h, y0 + max(1, int(rh * h)))
        image = image[y0:y1, x0:x1]
    return np.ascontiguousarray(image)
