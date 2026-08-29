"""The running feeder: one worker thread per bowl, one shared camera hub."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import UNKNOWN, __version__
from .actuators import ActuatorFactory
from .cameras import CameraHub, CameraView
from .config import AppConfig, BowlConfig
from .controller import BowlController
from .detector import Detector, build_detector
from .events import Event, EventLog
from .recognizer import Recognizer

log = logging.getLogger(__name__)

# A frame older than this means the camera has stalled; treat the bowl as unseen.
STALE_FRAME_S = 3.0
SNAPSHOT_MIN_INTERVAL_S = 5.0


class BowlWorker(threading.Thread):
    """Capture -> detect -> recognise -> state machine, at a fixed rate."""

    def __init__(
        self,
        cfg: BowlConfig,
        view: CameraView,
        detector: Detector,
        recognizer: Recognizer | None,
        controller: BowlController,
        loop_fps: float,
        snapshot_dir: Path | None = None,
    ):
        super().__init__(name=f"bowl-{cfg.id}", daemon=True)
        self.cfg = cfg
        self.view = view
        self.detector = detector
        self.recognizer = recognizer
        self.controller = controller
        self.period = 1.0 / max(loop_fps, 0.1)
        self.snapshot_dir = snapshot_dir
        self._stop = threading.Event()
        self._last_snapshot = 0.0
        self._last_state = controller.state
        self.latest_frame: np.ndarray | None = None
        self.latest_crop: np.ndarray | None = None
        self.frames = 0
        self.inferences = 0
        self.last_error: str | None = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info("%s: watching for %s", self.cfg.id, self.cfg.cat)
        last_present, last_label, last_conf = False, None, 0.0

        while not self._stop.is_set():
            started = time.monotonic()
            try:
                frame = self.view.read(only_new=True)
                if frame is None:
                    # No new image. Hold the previous observation rather than
                    # inventing an absence - unless the camera has clearly died.
                    stale = self.view.read()
                    if stale is None or time.monotonic() - stale.timestamp > STALE_FRAME_S:
                        last_present, last_label, last_conf = False, None, 0.0
                    self.controller.observe(last_present, last_label, last_conf)
                else:
                    self.frames += 1
                    self.latest_frame = frame.image
                    last_present, last_label, last_conf = self._process(frame.image)
                    self.controller.observe(last_present, last_label, last_conf)
                self.last_error = None
            except Exception as exc:  # keep one bad bowl from killing the rig
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("%s: worker iteration failed", self.cfg.id)
                time.sleep(0.5)

            self._maybe_snapshot()
            remaining = self.period - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

        log.info("%s: worker stopped", self.cfg.id)

    def _process(self, image: np.ndarray) -> tuple[bool, str | None, float]:
        detection = self.detector.detect(image)
        if detection is None:
            self.latest_crop = None
            return False, None, 0.0

        crop = detection.crop(image, pad_frac=0.15)
        self.latest_crop = crop
        if self.recognizer is None:
            return True, None, 0.0

        prediction = self.recognizer.predict(crop)
        self.inferences += 1
        return True, prediction.label, prediction.confidence

    def _maybe_snapshot(self) -> None:
        """Save the crop behind each state change - free extra training data."""
        if self.snapshot_dir is None or self.latest_crop is None:
            return
        state_changed = self.controller.state is not self._last_state
        self._last_state = self.controller.state
        now = time.monotonic()
        if not state_changed or now - self._last_snapshot < SNAPSHOT_MIN_INTERVAL_S:
            return
        self._last_snapshot = now
        try:
            import cv2

            label = self.controller.last_decision or UNKNOWN
            directory = self.snapshot_dir / label
            directory.mkdir(parents=True, exist_ok=True)
            name = f"{self.cfg.id}-{datetime.now():%Y%m%d-%H%M%S}-{self.controller.state.value}.jpg"
            cv2.imwrite(str(directory / name), self.latest_crop)
        except Exception:  # pragma: no cover - snapshots are never critical
            log.exception("%s: could not save snapshot", self.cfg.id)

    def status(self) -> dict:
        return {
            **self.controller.status(),
            "frames": self.frames,
            "inferences": self.inferences,
            "error": self.last_error,
        }


class FeederApp:
    def __init__(self, cfg: AppConfig, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.events = EventLog(cfg.log_dir)
        self.hub = CameraHub()
        self.factory = ActuatorFactory(cfg.actuator)
        self.workers: list[BowlWorker] = []
        self.started_at = time.time()
        self._status_server = None

    def build(self) -> None:
        recognizer = None
        if not self.dry_run:
            recognizer = Recognizer.from_config(self.cfg.recognition)
            log.info("recognising: %s", ", ".join(recognizer.bundle.labels))

        snapshot_dir = Path(self.cfg.snapshot_dir) if self.cfg.snapshot_dir else None
        for bowl in self.cfg.bowls:
            if not bowl.enabled:
                log.info("%s: disabled in config, skipping", bowl.id)
                continue
            actuator = self.factory.create(bowl.id, bowl.servo)
            actuator.close()   # known state before anything else happens
            controller = BowlController(
                bowl,
                actuator,
                vote_window=self.cfg.recognition.vote_window,
                votes_required=self.cfg.recognition.votes_required,
                on_event=self.events,
            )
            self.workers.append(
                BowlWorker(
                    cfg=bowl,
                    view=self.hub.view(bowl.camera),
                    detector=build_detector(self.cfg.detector),
                    recognizer=recognizer,
                    controller=controller,
                    loop_fps=self.cfg.loop_fps,
                    snapshot_dir=snapshot_dir,
                )
            )
        if not self.workers:
            raise RuntimeError("no enabled bowls in the config")

    def start(self) -> None:
        self.events.write(Event("startup", bowl="-", detail={"version": __version__,
                                                             "bowls": len(self.workers)}))
        for worker in self.workers:
            worker.start()
        if self.cfg.status_port:
            from .status import start_status_server

            self._status_server = start_status_server(self, int(self.cfg.status_port))

    def stop(self) -> None:
        log.info("shutting down")
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            if worker.ident is not None:      # never started: nothing to join
                worker.join(timeout=3.0)
        for worker in self.workers:
            try:
                worker.controller.force_close("shutdown")
            except Exception:  # pragma: no cover
                log.exception("%s: could not park the lid", worker.cfg.id)
        if self._status_server is not None:
            self._status_server.shutdown()
        self.hub.close()
        self.factory.shutdown()
        self.events.write(Event("shutdown", bowl="-", detail={"uptime_s": round(time.time() - self.started_at)}))

    def status(self) -> dict:
        return {
            "version": __version__,
            "uptime_s": round(time.time() - self.started_at),
            "dry_run": self.dry_run,
            "cameras": self.hub.healthy(),
            "bowls": [worker.status() for worker in self.workers],
            "recent_events": [
                {"time": datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S"),
                 "bowl": e.bowl, "kind": e.kind, "cat": e.cat, "detail": e.detail}
                for e in self.events.recent[-20:][::-1]
            ],
        }

    def run_forever(self) -> None:
        self.build()
        self.start()
        try:
            while any(worker.is_alive() for worker in self.workers):
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self.stop()
