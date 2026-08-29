"""Which cat is this, and are we sure enough to act on it?

Two layers of caution sit between the camera and the motor:

1. A per-frame confidence floor. Below it the frame is ``unknown``.
2. A sliding vote window. A lid only moves when several recent frames agree,
   which filters out the single bad frame caused by a blink, a yawn, or a tail
   swishing across the lens.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import UNKNOWN
from .config import RecognitionConfig
from .embedder import Embedder, EmbedderSpec, build_embedder

log = logging.getLogger(__name__)

OTHER = "_other"   # label for the optional negative class (other pets, humans, empty bowl)


class RecognizerError(RuntimeError):
    pass


@dataclass
class Prediction:
    label: str
    confidence: float
    raw_label: str = ""          # best class before the confidence floor was applied
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def is_known(self) -> bool:
        return self.label not in (UNKNOWN, OTHER)


@dataclass
class ClassifierBundle:
    """A trained classifier plus everything needed to reproduce its input."""

    model: Any
    labels: list[str]
    embedder: EmbedderSpec
    min_confidence: float = 0.75
    trained_at: float = field(default_factory=time.time)
    metrics: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("wrote classifier for %s to %s", ", ".join(self.labels), path)

    @staticmethod
    def load(path: str | Path) -> "ClassifierBundle":
        import joblib

        path = Path(path)
        if not path.exists():
            raise RecognizerError(
                f"no trained classifier at {path} - run 'catbowl train' first"
            )
        bundle = joblib.load(path)
        if not isinstance(bundle, ClassifierBundle):
            raise RecognizerError(f"{path} does not contain a catbowl classifier")
        return bundle


class Recognizer:
    def __init__(self, embedder: Embedder, bundle: ClassifierBundle, min_confidence: float | None = None):
        if embedder.spec != bundle.embedder:
            raise RecognizerError(
                "classifier/backbone mismatch - the classifier was trained with "
                f"{bundle.embedder} but the runtime is using {embedder.spec}. "
                "Retrain, or set recognition.backend/model to match."
            )
        self.embedder = embedder
        self.bundle = bundle
        self.min_confidence = bundle.min_confidence if min_confidence is None else min_confidence

    @classmethod
    def from_config(cls, cfg: RecognitionConfig) -> "Recognizer":
        return cls(build_embedder(cfg), ClassifierBundle.load(cfg.classifier), cfg.min_confidence)

    def predict(self, crop: np.ndarray) -> Prediction:
        return self.predict_batch([crop])[0]

    def predict_batch(self, crops: list[np.ndarray]) -> list[Prediction]:
        vectors = self.embedder.embed_batch(crops)
        probabilities = self.bundle.model.predict_proba(vectors)
        labels = list(self.bundle.model.classes_)

        out = []
        for row in probabilities:
            best = int(np.argmax(row))
            raw, confidence = labels[best], float(row[best])
            label = raw if confidence >= self.min_confidence else UNKNOWN
            out.append(
                Prediction(
                    label=label,
                    confidence=confidence,
                    raw_label=raw,
                    probabilities={str(name): float(p) for name, p in zip(labels, row)},
                )
            )
        return out


class VoteTracker:
    """Sliding window over recent frames; reports a winner only on consensus."""

    def __init__(self, window: int, required: int):
        if required > window:
            raise ValueError("required votes cannot exceed the window size")
        self.window = window
        self.required = required
        self._votes: deque[str] = deque(maxlen=window)

    def update(self, label: str) -> None:
        self._votes.append(label)

    def clear(self) -> None:
        self._votes.clear()

    @property
    def votes(self) -> list[str]:
        return list(self._votes)

    def decision(self) -> str | None:
        """The label held by at least ``required`` of the last ``window`` frames."""
        if not self._votes:
            return None
        label, count = Counter(self._votes).most_common(1)[0]
        if count >= self.required and label not in (UNKNOWN, OTHER):
            return label
        return None

    def tally(self) -> dict[str, int]:
        return dict(Counter(self._votes))
