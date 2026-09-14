"""The running feeder: one worker thread per bowl, one shared camera hub."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import CROWD, UNKNOWN, __version__
from .actuators import ActuatorFactory
from .cameras import CameraHub, CameraView
from .config import AppConfig, BowlConfig, CaptureConfig
from .controller import BowlController
from .detector import Detector, build_detector
from .events import Event, EventLog
from .presort import VISIT_GAP_S, Shot, visit_verdict
from .recognizer import OTHER, Recognizer
from .sorting import DISCARD, Sorter

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
        capture: CaptureConfig | None = None,
    ):
        super().__init__(name=f"bowl-{cfg.id}", daemon=True)
        self.cfg = cfg
        self.view = view
        self.detector = detector
        self.recognizer = recognizer
        self.controller = controller
        self.period = 1.0 / max(loop_fps, 0.1)
        self.snapshot_dir = snapshot_dir
        self.capture = capture or CaptureConfig()
        # Photos taken during the current visit: (where it was filed, what the
        # classifier said). Emptied when the visit ends and is judged whole.
        self._visit: list[tuple[Path, Shot]] = []
        self._capture_root = Path(self.capture.dir) if self.capture.dir else None
        self._capture_dir = self._capture_root / "unsorted" if self._capture_root else None
        self._last_capture = 0.0
        # Seeded from what is already on disk so max_images caps the folder
        # rather than the run: otherwise a service that restarts nightly fills
        # the SD card a few thousand images at a time.
        self._captured = self._count_captures()
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
        rationed = [f"{cat} ({self.cfg.rations[cat].seconds:g}s per "
                    f"{self.cfg.rations[cat].per_s / 60:g}min)" if cat in self.cfg.rations else cat
                    for cat in self.cfg.cats]
        log.info("%s: watching for %s", self.cfg.id, ", ".join(rationed))
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

        # A visit interrupted by a shutdown still deserves its verdict: the
        # photos are already on disk, and judging them now saves a human the
        # frames the rest of the visit would have settled.
        self._settle_visit(force=True)
        log.info("%s: worker stopped", self.cfg.id)

    def _settle_visit(self, now: float | None = None, force: bool = False) -> int:
        """Re-file this visit's photos once the whole visit has been seen.

        Each photo is filed the moment it is taken, on its own evidence, so a
        crash or a kill never loses one. But a frame is not an independent
        sample: thirty captures two seconds apart are one cat, and a blurred
        one in the middle is the same animal as the twenty sure ones around it.
        So when the visit ends the frames are judged together (see presort.py)
        and any photo the visit overrules is moved to where the visit says it
        belongs - usually out of `unsure` and into a name.

        Returns how many photos were moved, for the tests and the log.
        """
        now = time.monotonic() if now is None else now
        if not self._visit:
            return 0
        if not force and now - self._last_capture < VISIT_GAP_S:
            return 0

        shots = [shot for _, shot in self._visit]
        verdict = visit_verdict(shots, self._recognizer_floor())
        # The model's "none of the cats" is filed under the name a human uses
        # for it, as _verdict_for does frame by frame - not in a folder of its own.
        label = DISCARD if verdict.label == OTHER else verdict.label
        moved = 0
        if label is not None:
            for path, shot in self._visit:
                if shot.verdict == label or not path.exists():
                    continue
                destination = self._capture_into(label)
                if destination is None:
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                try:
                    path.rename(destination / path.name)
                    moved += 1
                except OSError:  # pragma: no cover - a human may have filed it already
                    log.debug("%s: could not re-file %s", self.cfg.id, path.name)
        if moved:
            log.info("%s: visit of %d photos settled as %s, %d re-filed",
                     self.cfg.id, len(shots), label, moved)
        self._visit.clear()
        return moved

    def _recognizer_floor(self) -> float:
        return self.recognizer.min_confidence if self.recognizer else 1.0

    def _process(self, image: np.ndarray) -> tuple[bool, str | None, float]:
        detection = self.detector.detect(image)
        if detection is None:
            self.latest_crop = None
            self._settle_visit()
            return False, None, 0.0

        crop = detection.crop(image, pad_frac=0.15)
        self.latest_crop = crop
        if detection.crowd > 1:
            # Two cats at one bowl: whichever one the crop shows, the other is
            # standing right there, so an open lid feeds the wrong cat. Answered
            # before the classifier runs - the question "which cat is this" has
            # no useful answer here, and it is the one question it can answer.
            # Banked unsorted: a photo of two cats is not a photo of either of
            # them, and no proposal tab is the right home for it.
            self._maybe_capture(image, crop)
            return True, CROWD, 1.0
        if self.recognizer is None:
            # No classifier: identity is stubbed out and anything the detector
            # finds is treated as this bowl's own cat. Everything downstream -
            # vote window, confirmation timer, slew, close delay - still runs,
            # so this exercises the whole rig before a model exists. It will
            # also open the lid for the wrong cat, a hand, or a passing dog.
            self._maybe_capture(image, crop)
            return True, self.cfg.cat, 1.0

        prediction = self.recognizer.predict(crop)
        self.inferences += 1
        verdict = self._verdict_for(prediction)
        banked = self._maybe_capture(image, crop, verdict)
        if banked is not None:
            self._remember(banked, verdict, prediction)
        return True, prediction.label, prediction.confidence

    def _remember(self, path: Path, verdict: str, prediction) -> None:
        """Keep this photo's verdict until the visit it belongs to has ended."""
        self._visit.append((path, Shot(
            name=path.name,
            taken=time.monotonic(),
            probabilities=dict(prediction.probabilities),
            label=prediction.raw_label,
            confidence=prediction.confidence,
            verdict=verdict,
        )))

    def _verdict_for(self, prediction) -> str:
        """The proposal folder a prediction belongs in."""
        from .recognizer import OTHER
        from .sorting import DISCARD

        if prediction.raw_label == OTHER and prediction.confidence >= self.recognizer.min_confidence:
            return DISCARD           # "none of the cats", filed where those live
        return prediction.label if prediction.is_known else "unsure"

    def _count_captures(self) -> int:
        """Photos this rig has banked and nobody has filed yet.

        The queue and the machine's proposals both count: max_images is there
        to keep the SD card alive, and an unchecked proposal takes up exactly
        as much of it as an unsorted photo.
        """
        if self._capture_root is None:
            return 0
        total = 0
        for directory in (self._capture_dir, *sorted((self._capture_root / "proposed").glob("*"))):
            if directory is not None and directory.is_dir():
                total += sum(1 for _ in directory.glob("*.jpg"))
        return total

    def _capture_into(self, verdict: str | None) -> Path | None:
        """Where this photo should be filed.

        With no classifier, or none that was sure, the photo joins the queue a
        human works through. With one, it goes to `proposed/<its guess>` - the
        same place `catbowl presort` writes, browsable in the same tabs, and
        accepted or corrected with the same click. The rig keeps its opinions
        out of the folders a person has vouched for, which is the only rule
        that matters here: a wrong guess filed as fact is a wrong label, and
        the next model learns it.
        """
        if self._capture_root is None:
            return None
        if verdict is None:
            return self._capture_dir
        return self._capture_root / "proposed" / verdict

    def _maybe_capture(self, frame: np.ndarray, crop: np.ndarray,
                       verdict: str | None = None) -> Path | None:
        """Bank a photo of whatever is at the bowl, for a later training run.

        Runs on every detection rather than on state changes, because the point
        is a varied dataset: a cat mid-turn or head-down is exactly the pose the
        classifier gets wrong, and those frames never coincide with a lid moving.
        """
        directory = self._capture_into(verdict)
        if directory is None:
            return None
        now = time.monotonic()
        if now - self._last_capture < self.capture.interval_s:
            return None
        if self.capture.max_images and self._captured >= self.capture.max_images:
            return None
        self._last_capture = now
        try:
            import cv2

            directory.mkdir(parents=True, exist_ok=True)
            stamp = f"{datetime.now():%Y%m%d-%H%M%S-%f}"[:-3]
            path = directory / f"{self.cfg.id}-{stamp}.jpg"
            cv2.imwrite(str(path), crop)
            if self.capture.save_frame:
                cv2.imwrite(str(directory / f"{self.cfg.id}-{stamp}-frame.jpg"), frame)
            self._captured += 1
            if self.capture.max_images and self._captured == self.capture.max_images:
                log.info("%s: capture folder has reached max_images (%d); stopping",
                         self.cfg.id, self.capture.max_images)
            return path
        except Exception:  # pragma: no cover - collecting data is never critical
            log.exception("%s: could not save a capture", self.cfg.id)
            return None

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
            "captured": self._captured,
            "inferences": self.inferences,
            "error": self.last_error,
        }


class FeederApp:
    def __init__(self, cfg: AppConfig, no_model: bool = False):
        self.cfg = cfg
        self.no_model = no_model
        self.events = EventLog(cfg.log_dir)
        self.hub = CameraHub()
        self.factory = ActuatorFactory(cfg.actuator)
        self.workers: list[BowlWorker] = []
        # Only exists when photos are being captured; /sort has nothing to show
        # otherwise.
        self.sorter = (
            Sorter(cfg.capture.dir, cfg.capture.labels) if cfg.capture.dir else None
        )
        self.started_at = time.time()
        self._status_server = None

    def build(self) -> None:
        recognizer = None
        if self.no_model:
            log.warning(
                "no-model: every detection counts as the bowl's own cat. "
                "Any cat, hand or dog will open the lid. Testing only."
            )
        else:
            recognizer = Recognizer.from_config(self.cfg.recognition)
            log.info("recognising: %s", ", ".join(recognizer.bundle.labels))

        snapshot_dir = Path(self.cfg.snapshot_dir) if self.cfg.snapshot_dir else None
        for bowl in self.cfg.bowls:
            if not bowl.enabled:
                log.info("%s: disabled in config, skipping", bowl.id)
                continue
            actuator = self.factory.create(bowl.id, bowl.servos)
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
                    capture=self.cfg.capture,
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
            "no_model": self.no_model,
            "cameras": self.hub.healthy(),
            "bowls": [worker.status() for worker in self.workers],
            "recent_events": [
                {"time": datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S"),
                 "bowl": e.bowl, "kind": e.kind, "cat": e.cat, "detail": e.detail}
                for e in self.events.recent[-20:][::-1]
            ],
        }

    def set_manual(self, bowl_id: str, mode: str | None) -> None:
        """Pin one bowl's lid by hand. Raises KeyError for an unknown bowl."""
        worker = next((w for w in self.workers if w.cfg.id == bowl_id), None)
        if worker is None:
            raise KeyError(bowl_id)
        log.info("%s: manual override -> %s", bowl_id, mode or "auto")
        worker.controller.set_manual(mode)

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
