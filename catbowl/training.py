"""Dataset building, training and evaluation.

The workflow is deliberately boring:

    photos/ or cameras  ->  crops/<cat>/*.jpg  ->  embeddings  ->  classifier

Because the backbone is frozen, training is a logistic regression over a few
hundred vectors: seconds on a laptop, and quick enough on the Pi itself.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DetectorConfig, RecognitionConfig
from .detector import Detector, build_detector
from .embedder import Embedder, build_embedder
from .recognizer import OTHER, ClassifierBundle

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_PER_CLASS = 15


@dataclass
class Dataset:
    paths: list[Path]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.labels))

    def counts(self) -> dict[str, int]:
        return {label: self.labels.count(label) for label in self.classes}


def load_dataset(root: str | Path) -> Dataset:
    """Read ``root/<label>/*.jpg`` into a flat list."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    paths, labels = [], []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        for image in sorted(directory.iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(image)
                labels.append(directory.name)
    if not paths:
        raise FileNotFoundError(
            f"no images under {root} - expected one subdirectory per cat, e.g. {root}/mochi/*.jpg"
        )
    return Dataset(paths, labels)


# --------------------------------------------------------------------------- #
# Building crops
# --------------------------------------------------------------------------- #

def import_photos(
    source: str | Path,
    label: str,
    out_root: str | Path,
    detector: Detector,
    keep_uncropped: bool = False,
    min_size: int = 64,
    recursive: bool = True,
) -> dict[str, int]:
    """Detect the cat in each photo under *source* and save a tight crop.

    Phone photos are usually much larger than the camera frames the model will
    see at runtime, but since we crop to the animal and resize to a fixed input,
    the two end up in the same place.
    """
    import cv2

    source, out_root = Path(source), Path(out_root)
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    files = [p for p in sorted(source.glob(pattern)) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"no images found in {source}")

    stats = {"total": len(files), "saved": 0, "no_detection": 0, "unreadable": 0, "too_small": 0}
    for index, path in enumerate(files):
        image = cv2.imread(str(path))
        if image is None:
            stats["unreadable"] += 1
            continue
        # Big phone photos: shrink before detection, it changes nothing but speed.
        scale = 1024 / max(image.shape[:2])
        work = cv2.resize(image, (0, 0), fx=scale, fy=scale) if scale < 1 else image

        detection = detector.detect(work)
        if detection is None:
            stats["no_detection"] += 1
            if not keep_uncropped:
                continue
            crop = work
        else:
            crop = detection.crop(work, pad_frac=0.15)

        if min(crop.shape[:2]) < min_size:
            stats["too_small"] += 1
            continue
        cv2.imwrite(str(out_dir / f"{label}-{index:05d}{path.suffix.lower()}"), crop)
        stats["saved"] += 1
        if (index + 1) % 25 == 0:
            log.info("%s: %d/%d processed", label, index + 1, len(files))
    return stats


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def embed_dataset(
    dataset: Dataset,
    embedder: Embedder,
    augment: bool = True,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed every image, optionally adding a mirrored copy of each one."""
    import cv2

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    batch: list[np.ndarray] = []
    batch_labels: list[str] = []

    def flush() -> None:
        if batch:
            vectors.append(embedder.embed_batch(batch))
            labels.extend(batch_labels)
            batch.clear()
            batch_labels.clear()

    for i, (path, label) in enumerate(zip(dataset.paths, dataset.labels)):
        image = cv2.imread(str(path))
        if image is None:
            log.warning("skipping unreadable image %s", path)
            continue
        batch.append(image)
        batch_labels.append(label)
        if augment:
            batch.append(cv2.flip(image, 1))
            batch_labels.append(label)
        if len(batch) >= batch_size:
            flush()
        if (i + 1) % 100 == 0:
            log.info("embedded %d/%d images", i + 1, len(dataset))
    flush()

    if not vectors:
        raise RuntimeError("no images could be embedded")
    return np.concatenate(vectors), np.array(labels)


def suggest_threshold(
    probabilities: np.ndarray,
    classes: list[str],
    truth: np.ndarray,
    target_precision: float = 0.99,
) -> tuple[float, list[dict]]:
    """Lowest confidence floor that still hits *target_precision* on held-out data.

    A high floor means fewer wrong lids opening but more "the bowl ignored me"
    moments; the sweep is printed so the trade-off is a choice, not a guess.
    """
    predicted = np.array([classes[i] for i in probabilities.argmax(axis=1)])
    confidence = probabilities.max(axis=1)

    sweep = []
    for threshold in np.arange(0.30, 0.99, 0.02):
        accepted = confidence >= threshold
        n = int(accepted.sum())
        correct = int((predicted[accepted] == truth[accepted]).sum()) if n else 0
        sweep.append({
            "threshold": round(float(threshold), 2),
            "coverage": round(n / len(truth), 3),
            "precision": round(correct / n, 4) if n else 1.0,
            "accepted": n,
        })

    good = [row for row in sweep if row["precision"] >= target_precision and row["coverage"] > 0.4]
    chosen = good[0]["threshold"] if good else 0.75
    return float(chosen), sweep


def train(
    data_root: str | Path,
    recognition: RecognitionConfig,
    out_path: str | Path | None = None,
    test_size: float = 0.25,
    augment: bool = True,
    seed: int = 0,
    target_precision: float = 0.99,
) -> tuple[ClassifierBundle, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    random.seed(seed)
    np.random.seed(seed)

    dataset = load_dataset(data_root)
    counts = dataset.counts()
    log.info("dataset: %d images across %s", len(dataset), counts)

    real_classes = [c for c in dataset.classes if c != OTHER]
    if len(real_classes) < 2:
        raise ValueError(
            f"need at least two cats to tell apart, found {real_classes}. "
            f"Add one subdirectory per cat under {data_root}."
        )
    thin = {k: v for k, v in counts.items() if v < MIN_PER_CLASS}
    if thin:
        log.warning("thin classes (fewer than %d images): %s - expect shaky accuracy",
                    MIN_PER_CLASS, thin)

    embedder = build_embedder(recognition)
    log.info("embedding with %s", embedder.spec)
    X, y = embed_dataset(dataset, embedder, augment=augment)

    stratify = y if min(counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    model = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
    model.fit(X_train, y_train)

    classes = list(model.classes_)
    probabilities = model.predict_proba(X_test)
    predicted = np.array([classes[i] for i in probabilities.argmax(axis=1)])
    threshold, sweep = suggest_threshold(probabilities, classes, y_test, target_precision)

    metrics = {
        "n_images": len(dataset),
        "n_vectors": int(len(X)),
        "counts": counts,
        "raw_accuracy": float((predicted == y_test).mean()),
        "report": classification_report(y_test, predicted, zero_division=0, output_dict=True),
        "confusion": confusion_matrix(y_test, predicted, labels=classes).tolist(),
        "classes": classes,
        "threshold_sweep": sweep,
        "suggested_threshold": threshold,
    }

    bundle = ClassifierBundle(
        model=model,
        labels=classes,
        embedder=embedder.spec,
        min_confidence=threshold,
        metrics=metrics,
    )
    if out_path:
        bundle.save(out_path)
    return bundle, metrics


def format_report(metrics: dict) -> str:
    """Human-readable training summary for the terminal."""
    classes = metrics["classes"]
    width = max(len(c) for c in classes) + 2
    lines = [
        f"images: {metrics['n_images']}  vectors: {metrics['n_vectors']}  "
        f"raw accuracy: {metrics['raw_accuracy']:.1%}",
        "",
        "confusion matrix (rows = truth, columns = predicted)",
        " " * width + "".join(c.rjust(width) for c in classes),
    ]
    for name, row in zip(classes, metrics["confusion"]):
        lines.append(name.rjust(width) + "".join(str(v).rjust(width) for v in row))

    lines += ["", "confidence threshold sweep", "  thresh  coverage  precision"]
    for row in metrics["threshold_sweep"]:
        if round(row["threshold"] * 100) % 10 == 0:
            lines.append(f"  {row['threshold']:>6.2f}  {row['coverage']:>8.1%}  {row['precision']:>9.1%}")
    lines += ["", f"suggested recognition.min_confidence: {metrics['suggested_threshold']:.2f}"]

    per_class = metrics["report"]
    weak = [c for c in classes if per_class.get(c, {}).get("recall", 1.0) < 0.8]
    if weak:
        lines.append(f"weak recall for: {', '.join(weak)} - more photos of these would help")
    return "\n".join(lines)


def build_import_detector(detector_type: str = "ssdlite") -> Detector:
    """Detector used when importing stills (motion subtraction is useless there)."""
    if detector_type == "motion":
        log.warning("motion detection cannot work on unrelated still photos; using 'none'")
        detector_type = "none"
    return build_detector(DetectorConfig(type=detector_type))
