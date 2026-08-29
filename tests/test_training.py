import numpy as np
import pytest

from catbowl.config import RecognitionConfig
from catbowl.detector import DetectorConfig, MotionDetector, NullDetector
from catbowl.recognizer import Recognizer
from catbowl.training import (
    format_report,
    import_photos,
    load_dataset,
    suggest_threshold,
    train,
)


COLOURS = {"ginger": (40, 120, 220), "tuxedo": (30, 30, 30), "tabby": (120, 150, 160)}


def write_cats(root, per_cat=24, size=64):
    """A toy dataset: one solid colour per 'cat', plus noise and a shared background."""
    import cv2

    rng = np.random.default_rng(7)
    for label, colour in COLOURS.items():
        directory = root / label
        directory.mkdir(parents=True)
        for i in range(per_cat):
            image = rng.integers(45, 65, (size, size, 3)).astype(np.int16)
            image[12:52, 12:52] = np.array(colour) + rng.integers(-25, 25, 3)
            cv2.imwrite(str(directory / f"{i:03d}.jpg"), np.clip(image, 0, 255).astype(np.uint8))
    return root


def test_load_dataset_reads_one_directory_per_cat(tmp_path):
    dataset = load_dataset(write_cats(tmp_path / "crops"))
    assert dataset.classes == sorted(COLOURS)
    assert dataset.counts() == {label: 24 for label in COLOURS}
    assert len(dataset) == 72


def test_empty_dataset_directory_is_a_clear_error(tmp_path):
    empty = tmp_path / "crops"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="one subdirectory per cat"):
        load_dataset(empty)


def test_absent_dataset_directory_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset directory not found"):
        load_dataset(tmp_path / "does-not-exist")


def test_training_separates_the_classes(tmp_path):
    root = write_cats(tmp_path / "crops")
    cfg = RecognitionConfig(backend="mock")
    bundle, metrics = train(root, cfg, out_path=tmp_path / "classifier.joblib", augment=False)

    assert set(bundle.labels) == set(COLOURS)
    assert metrics["raw_accuracy"] > 0.95
    assert 0.3 <= bundle.min_confidence <= 0.99
    assert (tmp_path / "classifier.joblib").exists()
    assert "confusion matrix" in format_report(metrics)


def test_a_trained_bundle_drives_the_recogniser(tmp_path):
    import cv2

    from catbowl.embedder import build_embedder

    root = write_cats(tmp_path / "crops")
    cfg = RecognitionConfig(backend="mock", min_confidence=0.5)
    bundle, _ = train(root, cfg, out_path=tmp_path / "classifier.joblib", augment=False)

    recognizer = Recognizer(build_embedder(cfg), bundle, min_confidence=0.5)
    sample = cv2.imread(str(next((root / "tuxedo").iterdir())))
    assert recognizer.predict(sample).label == "tuxedo"


def test_augmentation_doubles_the_vectors(tmp_path):
    root = write_cats(tmp_path / "crops", per_cat=16)
    cfg = RecognitionConfig(backend="mock")
    _, plain = train(root, cfg, augment=False)
    _, mirrored = train(root, cfg, augment=True)
    assert mirrored["n_vectors"] == 2 * plain["n_vectors"]


def test_one_cat_is_not_a_classification_problem(tmp_path):
    import cv2

    root = tmp_path / "crops" / "only"
    root.mkdir(parents=True)
    for i in range(5):
        cv2.imwrite(str(root / f"{i}.jpg"), np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(ValueError, match="at least two cats"):
        train(tmp_path / "crops", RecognitionConfig(backend="mock"))


def test_threshold_sweep_trades_coverage_for_precision():
    classes = ["a", "b"]
    truth = np.array(["a"] * 50 + ["b"] * 50)
    confident = np.array([[0.95, 0.05]] * 50 + [[0.05, 0.95]] * 50)
    unsure = np.array([[0.55, 0.45]] * 50 + [[0.45, 0.55]] * 50)
    # Half the set is correct but unsure, half is confident and wrong.
    probabilities = np.vstack([confident[:50], unsure[50:]])
    threshold, sweep = suggest_threshold(probabilities, classes, truth)
    assert 0.3 <= threshold <= 0.99
    coverage = [row["coverage"] for row in sweep]
    assert coverage == sorted(coverage, reverse=True), "coverage falls as the bar rises"


def test_import_photos_crops_and_reports(tmp_path):
    import cv2

    source = tmp_path / "phone"
    source.mkdir()
    for i in range(4):
        cv2.imwrite(str(source / f"IMG_{i}.jpg"),
                    np.full((200, 200, 3), 90, np.uint8))
    stats = import_photos(source, "mochi", tmp_path / "crops", NullDetector())
    assert stats["saved"] == 4
    assert len(list((tmp_path / "crops" / "mochi").iterdir())) == 4


def test_import_skips_unreadable_files(tmp_path):
    source = tmp_path / "phone"
    source.mkdir()
    (source / "broken.jpg").write_text("this is not a jpeg")
    stats = import_photos(source, "mochi", tmp_path / "crops", NullDetector())
    assert stats["unreadable"] == 1 and stats["saved"] == 0


def test_motion_detector_ignores_a_static_scene():
    detector = MotionDetector(DetectorConfig(warmup_frames=5, min_area_frac=0.02))
    background = np.full((240, 320, 3), 60, np.uint8)
    for _ in range(20):
        detector.detect(background)
    assert detector.detect(background) is None


def test_motion_detector_finds_an_intruding_blob():
    detector = MotionDetector(DetectorConfig(warmup_frames=5, min_area_frac=0.01))
    background = np.full((240, 320, 3), 60, np.uint8)
    for _ in range(30):
        detector.detect(background)
    frame = background.copy()
    frame[80:180, 100:220] = 230
    detection = None
    for _ in range(3):
        detection = detector.detect(frame) or detection
    assert detection is not None
    x, y, w, h = detection.bbox
    assert w > 50 and h > 40
    assert detection.crop(frame).shape[2] == 3
