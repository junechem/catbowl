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


def make_app(workdir: Path, model_path: Path, **overrides) -> FeederApp:
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
    return FeederApp(cfg)


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
