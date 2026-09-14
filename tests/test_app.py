"""End-to-end: synthetic cameras, a real detector and classifier, simulated lids."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from catbowl.app import FeederApp
from catbowl.cameras import SyntheticCapture
from catbowl.config import DetectorConfig, RecognitionConfig, build_config
from catbowl.detector import MotionDetector
from catbowl.training import train

CATS = ["alpha", "bravo"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train the mock backbone on crops taken from the synthetic cameras."""
    workdir = tmp_path_factory.mktemp("catbowl")
    crops = workdir / "crops"
    for variant, cat in enumerate(CATS):
        (crops / cat).mkdir(parents=True)
        camera = SyntheticCapture(320, 240, fps=100000, variant=variant)
        # Build the dataset through the same detector the runtime uses, so the
        # crops the classifier trains on match the crops it will be shown.
        detector = MotionDetector(DetectorConfig(warmup_frames=3, min_area_frac=0.01))
        saved = 0
        for _ in range(4000):
            if saved >= 24:
                break
            frame = camera.read()
            detection = detector.detect(frame)
            if detection is None:
                continue
            crop = detection.crop(frame, pad_frac=0.15)
            if min(crop.shape[:2]) < 16:
                continue
            noise = np.random.default_rng(saved).integers(-10, 10, crop.shape)
            cv2.imwrite(str(crops / cat / f"{saved:03d}.jpg"),
                        np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8))
            saved += 1
        assert saved == 24, f"only produced {saved} crops for {cat}"

    model_path = workdir / "classifier.joblib"
    _, metrics = train(crops, RecognitionConfig(backend="mock"), out_path=model_path, augment=False)
    assert metrics["raw_accuracy"] > 0.95
    return workdir, model_path


def make_app(workdir: Path, model_path: Path, no_model: bool = False, **overrides) -> FeederApp:
    cfg = build_config({
        "recognition": {"backend": "mock", "classifier": str(model_path),
                        "min_confidence": 0.6, "vote_window": 4, "votes_required": 3},
        "detector": {"type": "motion", "warmup_frames": 3, "min_area_frac": 0.01},
        "actuator": {"driver": "mock"},
        "loop_fps": 20,
        "status_port": None,
        "log_dir": str(workdir / "logs"),
        "bowls": [
            {"id": f"bowl{i + 1}", "cat": cat,
             "camera": {"device": f"synthetic:{i}", "width": 320, "height": 240, "fps": 40},
             "servo": {"channel": i},
             "policy": {"open_confirm_s": 0.2, "close_delay_s": 0.6,
                        "cooldown_s": 0.3, "max_open_s": 30}}
            for i, cat in enumerate(CATS)
        ],
        **overrides,
    })
    return FeederApp(cfg, no_model=no_model)


def run_for(app: FeederApp, seconds: float) -> None:
    app.build()
    app.start()
    try:
        time.sleep(seconds)
    finally:
        app.stop()


def test_each_bowl_opens_for_its_own_cat(trained):
    workdir, model_path = trained
    app = make_app(workdir, model_path)
    run_for(app, 8.0)

    for worker in app.workers:
        assert worker.last_error is None, worker.last_error
        assert worker.frames > 20, "the camera thread should have delivered frames"
        assert worker.controller.stats["opens"] > 0, f"{worker.cfg.id} never opened for {worker.cfg.cat}"


def test_lids_are_parked_closed_on_shutdown(trained):
    workdir, model_path = trained
    app = make_app(workdir, model_path)
    run_for(app, 5.0)
    for worker in app.workers:
        assert worker.controller.actuator.position == 0.0


def test_events_are_written_to_disk(trained):
    workdir, model_path = trained
    app = make_app(workdir, model_path)
    run_for(app, 6.0)

    files = list((workdir / "logs").glob("events-*.jsonl"))
    assert files, "an event log should have been created"
    lines = files[0].read_text().strip().splitlines()
    kinds = {__import__("json").loads(line)["kind"] for line in lines}
    assert "startup" in kinds and "opened" in kinds


def test_status_payload_is_serialisable(trained):
    workdir, model_path = trained
    app = make_app(workdir, model_path)
    app.build()
    app.start()
    try:
        time.sleep(2.0)
        payload = __import__("json").loads(__import__("json").dumps(app.status(), default=str))
        assert len(payload["bowls"]) == len(CATS)
        assert set(payload["cameras"].values()) == {True}
    finally:
        app.stop()


def test_a_disabled_bowl_is_skipped(trained):
    workdir, model_path = trained
    app = make_app(workdir, model_path)
    app.cfg.bowls[1].enabled = False
    app.build()
    try:
        assert [w.cfg.id for w in app.workers] == ["bowl1"]
    finally:
        app.stop()


# --------------------------------------------------------------------------- #
# --no-model: identity stubbed out
# --------------------------------------------------------------------------- #

def test_no_model_treats_any_detection_as_the_bowls_own_cat():
    """Without a classifier the rig still has to open, or the flag is useless."""
    from catbowl.app import BowlWorker
    from catbowl.config import BowlConfig, ServoConfig

    bowl = BowlConfig(id="bowl1", cat="mochi", servos=[ServoConfig(channel=0)])
    worker = BowlWorker.__new__(BowlWorker)      # no thread, no camera
    worker.cfg = bowl
    worker.recognizer = None
    worker.latest_crop = None
    worker._capture_dir = None                   # no dataset collection here
    worker._capture_root = None
    worker._visit = []
    worker._last_capture = 0.0
    worker.detector = MotionDetector(DetectorConfig(warmup_frames=1, min_area_frac=0.001))

    # Learn an empty scene, then put something in it.
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(4):
        worker.detector.detect(blank)
    frame = blank.copy()
    frame[30:90, 40:120] = 255

    present, label, confidence = worker._process(frame)
    assert present
    assert label == "mochi", "the detection must be attributed to this bowl's cat"
    assert confidence == 1.0


def test_a_recognizer_free_worker_reports_nothing_when_nothing_moves():
    from catbowl.app import BowlWorker
    from catbowl.config import BowlConfig, ServoConfig

    worker = BowlWorker.__new__(BowlWorker)
    worker.cfg = BowlConfig(id="bowl1", cat="mochi", servos=[ServoConfig(channel=0)])
    worker.recognizer = None
    worker.latest_crop = None
    worker._capture_dir = None                   # no dataset collection here
    worker._capture_root = None
    worker._visit = []
    worker._last_capture = 0.0
    worker.detector = MotionDetector(DetectorConfig(warmup_frames=1, min_area_frac=0.001))

    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(4):
        worker.detector.detect(blank)
    assert worker._process(blank) == (False, None, 0.0)


def test_detections_are_banked_for_later_labelling(trained):
    """The dataset builds itself while the rig runs.

    With a classifier loaded the photos land in `proposed/<its guess>`, the
    same folders `catbowl presort` writes and the same tabs a human checks
    them in. Never in the folders a person has vouched for.
    """
    workdir, model_path = trained
    collected = workdir / "collected"
    app = make_app(workdir, model_path,
                   capture={"dir": str(collected), "interval_s": 0.2})
    run_for(app, 5.0)

    images = list((collected / "proposed").glob("*/*.jpg"))
    assert images, "nothing was captured"
    assert not list(collected.glob("*.jpg")), "nothing may land outside a bucket"
    for cat in (w.cfg.cat for w in app.workers):
        assert not (collected / cat).exists(), "the rig must not file into a human's folder"
    assert {p.name.split("-")[0] for p in images} == {w.cfg.id for w in app.workers}
    assert sum(w.status()["captured"] for w in app.workers) == len(images)


def test_capture_stops_at_max_images(trained):
    workdir, model_path = trained
    collected = workdir / "capped"
    app = make_app(workdir, model_path,
                   capture={"dir": str(collected), "interval_s": 0.05, "max_images": 2})
    run_for(app, 4.0)

    images = list((collected / "proposed").glob("*/*.jpg"))
    assert 0 < len(images) <= 2 * len(app.workers)


# --------------------------------------------------------------------------- #
# two cats at one bowl
# --------------------------------------------------------------------------- #

def _crowd_worker(recognizer, crowd):
    """A worker whose detector always sees *crowd* cats in one box."""
    from catbowl.app import BowlWorker
    from catbowl.config import BowlConfig, ServoConfig
    from catbowl.detector import Detection, Detector

    class Always(Detector):
        def detect(self, image):
            return Detection((10, 10, 40, 40), 0.9, "ssdlite", crowd=crowd)

    worker = BowlWorker.__new__(BowlWorker)
    worker.cfg = BowlConfig(id="bowl1", cat="mochi", servos=[ServoConfig(channel=0)])
    worker.recognizer = recognizer
    worker.latest_crop = None
    worker._capture_dir = None
    worker._capture_root = None
    worker._visit = []
    worker._last_capture = 0.0
    worker.inferences = 0
    worker.detector = Always()
    return worker


class Sure:
    """A classifier that is certain every crop is mochi. Counts its calls."""

    def __init__(self):
        self.calls = 0

    def predict(self, crop):
        from catbowl.recognizer import Prediction

        self.calls += 1
        return Prediction(label="mochi", confidence=1.0, raw_label="mochi")


def test_two_cats_are_never_reported_as_the_bowls_own_cat():
    """The crop shows one of them; the other is standing right beside it."""
    from catbowl import CROWD

    worker = _crowd_worker(Sure(), crowd=2)
    present, label, _ = worker._process(np.zeros((120, 160, 3), dtype=np.uint8))
    assert present
    assert label == CROWD
    assert worker.recognizer.calls == 0, "identity is not a question worth asking here"


def test_two_cats_are_refused_even_with_no_classifier_at_all():
    from catbowl import CROWD

    worker = _crowd_worker(None, crowd=2)
    assert worker._process(np.zeros((120, 160, 3), dtype=np.uint8))[1] == CROWD


def test_one_cat_still_reaches_the_classifier():
    worker = _crowd_worker(Sure(), crowd=1)
    assert worker._process(np.zeros((120, 160, 3), dtype=np.uint8))[1] == "mochi"


def test_without_a_classifier_captures_still_go_to_the_queue(trained):
    """--no-model has no opinion to file under, so a human gets all of them."""
    workdir, model_path = trained
    collected = workdir / "nomodel"
    app = make_app(workdir, model_path,
                   capture={"dir": str(collected), "interval_s": 0.2}, no_model=True)
    run_for(app, 4.0)

    assert list((collected / "unsorted").glob("*.jpg"))
    assert not (collected / "proposed").exists()


# --------------------------------------------------------------------------- #
# a visit is re-filed once it has been seen whole
# --------------------------------------------------------------------------- #

def _sorting_worker(tmp_path, floor=0.85):
    """A worker that banks photos, with the classifier stubbed out per call."""
    from catbowl.app import BowlWorker
    from catbowl.config import BowlConfig, CaptureConfig, ServoConfig

    worker = BowlWorker.__new__(BowlWorker)
    worker.cfg = BowlConfig(id="bowl1", cats=["J", "K"], servos=[ServoConfig(channel=0)])
    worker.capture = CaptureConfig(dir=str(tmp_path), interval_s=0.0)
    worker._capture_root = tmp_path
    worker._capture_dir = tmp_path / "unsorted"
    worker._visit = []
    worker._last_capture = 0.0
    worker._captured = 0
    worker.latest_crop = None
    worker.inferences = 0

    class Floor:
        min_confidence = floor

    worker.recognizer = Floor()
    return worker


def _bank(worker, verdict, probabilities, taken):
    """Pretend a photo was taken and filed under *verdict*."""
    from catbowl.presort import Shot

    directory = worker._capture_into(verdict)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"bowl1-2026091{len(worker._visit)}-120000-000.jpg"
    path.write_bytes(b"\xff\xd8jpeg")
    best = max(probabilities, key=lambda k: probabilities[k])
    worker._visit.append((path, Shot(name=path.name, taken=taken, probabilities=probabilities,
                                     label=best, confidence=probabilities[best],
                                     verdict=verdict)))
    worker._last_capture = taken
    return path


def test_a_blurred_frame_is_refiled_once_the_visit_has_been_seen(tmp_path):
    """The live half of the visit prior: unsure in the middle of a sure visit."""
    worker = _sorting_worker(tmp_path)
    sure = {"J": 0.97, "K": 0.02, "_other": 0.01}
    blur = {"J": 0.44, "K": 0.31, "_other": 0.25}
    for index, probabilities in enumerate([sure, sure, blur, sure, sure]):
        verdict = "J" if probabilities is sure else "unsure"
        stray = _bank(worker, verdict, probabilities, taken=1000.0 + index)

    moved = worker._settle_visit(now=1000.0 + 60, force=False)
    assert moved == 1
    assert not stray.exists() or stray.parent.name == "J"
    assert len(list((tmp_path / "proposed" / "J").glob("*.jpg"))) == 5
    assert not list((tmp_path / "proposed" / "unsure").glob("*.jpg"))


def test_a_visit_still_in_progress_is_left_alone(tmp_path):
    worker = _sorting_worker(tmp_path)
    _bank(worker, "J", {"J": 0.97, "K": 0.02, "_other": 0.01}, taken=time.monotonic())
    assert worker._settle_visit() == 0, "the cat is still at the bowl"
    assert worker._visit, "and its photos are still being collected"


def test_a_visit_of_two_cats_is_not_smoothed(tmp_path):
    """Same guard as presort: a shared bowl goes to a human, not to a majority."""
    worker = _sorting_worker(tmp_path)
    for index in range(3):
        _bank(worker, "J", {"J": 0.97, "K": 0.02, "_other": 0.01}, taken=1000.0 + index)
    for index in range(3, 5):
        _bank(worker, "K", {"J": 0.02, "K": 0.97, "_other": 0.01}, taken=1000.0 + index)

    assert worker._settle_visit(now=1100.0) == 0
    assert len(list((tmp_path / "proposed" / "J").glob("*.jpg"))) == 3
    assert len(list((tmp_path / "proposed" / "K").glob("*.jpg"))) == 2


def test_settling_a_visit_forgets_it(tmp_path):
    worker = _sorting_worker(tmp_path)
    _bank(worker, "J", {"J": 0.97, "K": 0.02, "_other": 0.01}, taken=1000.0)
    worker._settle_visit(now=1100.0)
    assert worker._visit == []


def test_a_visit_of_nothing_is_settled_into_discard(tmp_path):
    """The model says `_other`; the folder a human files those in is `discard`.

    Seen on the Pi on 2026-09-13: a visit settled as `_other` moved its photos
    into a `proposed/_other` folder of their own, beside `proposed/discard`.
    """
    worker = _sorting_worker(tmp_path)
    junk = {"J": 0.03, "K": 0.02, "_other": 0.95}
    blur = {"J": 0.30, "K": 0.25, "_other": 0.45}
    for index, probabilities in enumerate([junk, junk, blur, junk]):
        _bank(worker, "discard" if probabilities is junk else "unsure", probabilities,
              taken=1000.0 + index)

    assert worker._settle_visit(now=1100.0) == 1
    assert len(list((tmp_path / "proposed" / "discard").glob("*.jpg"))) == 4
    assert not (tmp_path / "proposed" / "_other").exists()
