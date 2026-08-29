import numpy as np
import pytest

from catbowl import UNKNOWN
from catbowl.config import RecognitionConfig
from catbowl.embedder import MockEmbedder, EmbedderSpec, build_embedder
from catbowl.recognizer import (
    OTHER,
    ClassifierBundle,
    Recognizer,
    RecognizerError,
    VoteTracker,
)


# -- voting ------------------------------------------------------------------ #

def test_no_votes_means_no_decision():
    assert VoteTracker(4, 3).decision() is None


def test_consensus_wins():
    tracker = VoteTracker(4, 3)
    for label in ["mochi", "pepper", "mochi", "mochi"]:
        tracker.update(label)
    assert tracker.decision() == "mochi"


def test_a_split_window_decides_nothing():
    tracker = VoteTracker(4, 3)
    for label in ["mochi", "pepper", "mochi", "pepper"]:
        tracker.update(label)
    assert tracker.decision() is None


def test_the_window_slides():
    tracker = VoteTracker(3, 3)
    for label in ["mochi"] * 3:
        tracker.update(label)
    assert tracker.decision() == "mochi"
    tracker.update("pepper")            # pushes the oldest mochi out
    assert tracker.decision() is None


@pytest.mark.parametrize("label", [UNKNOWN, OTHER])
def test_unknown_and_other_never_win(label):
    tracker = VoteTracker(3, 2)
    for _ in range(3):
        tracker.update(label)
    assert tracker.decision() is None


def test_clear_resets():
    tracker = VoteTracker(3, 2)
    tracker.update("mochi")
    tracker.update("mochi")
    tracker.clear()
    assert tracker.decision() is None and tracker.tally() == {}


def test_required_cannot_exceed_window():
    with pytest.raises(ValueError):
        VoteTracker(3, 4)


# -- embedding --------------------------------------------------------------- #

def test_mock_embeddings_are_unit_length_and_deterministic():
    embedder = MockEmbedder()
    image = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    first, second = embedder.embed(image), embedder.embed(image)
    assert np.allclose(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
    assert first.shape == (embedder.spec.dim,)


def test_embedder_separates_obviously_different_images():
    embedder = MockEmbedder()
    red = np.zeros((32, 32, 3), np.uint8); red[:, :, 2] = 220
    blue = np.zeros((32, 32, 3), np.uint8); blue[:, :, 0] = 220
    assert float(embedder.embed(red) @ embedder.embed(blue)) < 0.5


def test_build_embedder_honours_the_backend():
    assert isinstance(build_embedder(RecognitionConfig(backend="mock")), MockEmbedder)


# -- recognition ------------------------------------------------------------- #

class StubModel:
    classes_ = np.array(["mochi", "pepper"])

    def __init__(self, row):
        self.row = row

    def predict_proba(self, X):
        return np.tile(self.row, (len(X), 1))


def bundle_for(row, spec=None):
    return ClassifierBundle(
        model=StubModel(row),
        labels=["mochi", "pepper"],
        embedder=spec or MockEmbedder().spec,
        min_confidence=0.75,
    )


def test_confident_predictions_pass_through():
    recognizer = Recognizer(MockEmbedder(), bundle_for([0.95, 0.05]))
    prediction = recognizer.predict(np.zeros((32, 32, 3), np.uint8))
    assert prediction.label == "mochi" and prediction.is_known
    assert prediction.confidence == pytest.approx(0.95)


def test_low_confidence_becomes_unknown():
    recognizer = Recognizer(MockEmbedder(), bundle_for([0.55, 0.45]))
    prediction = recognizer.predict(np.zeros((32, 32, 3), np.uint8))
    assert prediction.label == UNKNOWN
    assert prediction.raw_label == "mochi", "the raw guess is kept for debugging"
    assert not prediction.is_known


def test_a_classifier_trained_on_another_backbone_is_refused():
    """Silently mixing backbones would produce confident nonsense."""
    wrong = EmbedderSpec("torch", "mobilenet_v3_small", 224, 576)
    with pytest.raises(RecognizerError, match="mismatch"):
        Recognizer(MockEmbedder(), bundle_for([0.9, 0.1], spec=wrong))


def test_missing_classifier_file_says_what_to_do(tmp_path):
    with pytest.raises(RecognizerError, match="catbowl train"):
        ClassifierBundle.load(tmp_path / "nope.joblib")


def test_bundle_round_trips(tmp_path):
    path = tmp_path / "classifier.joblib"
    bundle_for([0.9, 0.1]).save(path)
    loaded = ClassifierBundle.load(path)
    assert loaded.labels == ["mochi", "pepper"]
    assert loaded.embedder == MockEmbedder().spec
