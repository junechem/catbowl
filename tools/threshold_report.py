"""An honest score for the classifier, and what a confidence floor costs.

`catbowl train` splits its test set at random, one photo at a time. That
flatters the model twice over. The rig captures every two seconds, so a visit
leaves thirty near-identical frames and a random split puts some in training
and the rest in test - the model is asked about photos it has all but seen.
Augmentation runs before the split too, so a photo's mirror image can sit in
training while the original is tested.

This splits by *visit* instead: every frame of one approach is in training or
in test, never both, and no augmentation is used. The number it prints is what
the rig should expect from a cat it is seeing this minute, which is lower than
the training report and truer.

It then answers the question a threshold actually poses. Raising the floor
trades away frames the model would have got right for protection against the
ones it would have got wrong, and the two are not symmetrical here:

    wrong cat fed   - the bowl opened for the wrong animal
    cat turned away - the right cat was not recognised confidently, and waited
    junk let in     - an empty bowl or a blur was called a cat
    junk kept out   - correctly ignored

Usage:  .venv/bin/python tools/threshold_report.py [--config config/bowls.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catbowl.config import load_config                      # noqa: E402
from catbowl.embedder import build_embedder                 # noqa: E402
from catbowl.presort import VISIT_GAP_S, taken_at           # noqa: E402
from catbowl.recognizer import OTHER                        # noqa: E402
from catbowl.training import load_dataset                   # noqa: E402

CACHE = Path("~/.cache/catbowl-embeddings.npz").expanduser()


def visits_of(paths: list[Path], labels: list[str], gap_s: float = VISIT_GAP_S) -> np.ndarray:
    """A visit id per photo: consecutive captures of one cat, within *gap_s*."""
    order = sorted(range(len(paths)), key=lambda i: (labels[i], taken_at(paths[i].name) or 0.0))
    groups = np.zeros(len(paths), dtype=int)
    visit = 0
    previous_label, previous_time = None, None
    for i in order:
        when = taken_at(paths[i].name)
        if when is None or labels[i] != previous_label or previous_time is None \
                or when - previous_time > gap_s:
            visit += 1
        groups[i] = visit
        previous_label, previous_time = labels[i], when
    return groups


def embed(dataset, recognition, refresh: bool) -> np.ndarray:
    """Embed every photo, cached: the Pi takes seven minutes to do this."""
    names = np.array([str(p) for p in dataset.paths])
    if CACHE.exists() and not refresh:
        cached = np.load(CACHE, allow_pickle=True)
        if len(cached["names"]) == len(names) and (cached["names"] == names).all():
            print(f"using cached embeddings from {CACHE}")
            return cached["X"]

    import cv2

    embedder = build_embedder(recognition)
    print(f"embedding {len(names)} photos with {embedder.spec} (cached afterwards)")
    vectors, batch = [], []
    for index, path in enumerate(dataset.paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"unreadable: {path}")
        batch.append(image)
        if len(batch) >= 16:
            vectors.append(embedder.embed_batch(batch))
            batch = []
        if index % 500 == 0:
            print(f"  {index}/{len(names)}", flush=True)
    if batch:
        vectors.append(embedder.embed_batch(batch))
    X = np.concatenate(vectors)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, names=names)
    return X


def out_of_fold(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int):
    """Probabilities for every photo, from a model that never saw its visit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    classes = sorted(set(y))
    probabilities = np.zeros((len(y), len(classes)))
    for train_index, test_index in GroupKFold(n_splits=folds).split(X, y, groups):
        model = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
        model.fit(X[train_index], y[train_index])
        # Columns come back in the fold's own class order; map onto ours.
        column = {label: i for i, label in enumerate(model.classes_)}
        for label, target in zip(classes, range(len(classes))):
            if label in column:
                probabilities[test_index, target] = model.predict_proba(X[test_index])[:, column[label]]
    return classes, probabilities


def outcomes(truth: np.ndarray, predicted: np.ndarray, confidence: np.ndarray,
             threshold: float) -> dict:
    """The four things that can happen at a bowl, counted."""
    accepted = confidence >= threshold
    is_cat = truth != OTHER
    called_cat = predicted != OTHER
    return {
        "accepted": int(accepted.sum()),
        "total": len(truth),
        "right_cat_fed": int((accepted & is_cat & (predicted == truth)).sum()),
        "wrong_cat_fed": int((accepted & is_cat & called_cat & (predicted != truth)).sum()),
        "cat_turned_away": int((is_cat & (~accepted | (accepted & ~called_cat))).sum()),
        "junk_let_in": int((accepted & ~is_cat & called_cat).sum()),
        "junk_kept_out": int(((~is_cat) & (~accepted | ~called_cat)).sum()),
        "n_cats": int(is_cat.sum()),
        "n_junk": int((~is_cat).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/bowls.yaml")
    parser.add_argument("--data", default="data/collected")
    parser.add_argument("--labels", nargs="+", default=["J", "K", "F"])
    parser.add_argument("--negative", default="discard")
    parser.add_argument("--threshold", type=float, default=None,
                        help="the floor to report in detail (default: the config's)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--refresh", action="store_true", help="ignore the embedding cache")
    args = parser.parse_args()

    cfg = load_config(args.config)
    floor = args.threshold if args.threshold is not None else cfg.recognition.min_confidence

    dataset = load_dataset(args.data, args.labels, args.negative)
    y = np.array(dataset.labels)
    groups = visits_of(dataset.paths, dataset.labels)
    print(f"{len(y)} photos, {len(set(groups))} visits, {dict(zip(*np.unique(y, return_counts=True)))}")

    X = embed(dataset, cfg.recognition, args.refresh)
    classes, probabilities = out_of_fold(X, y, groups, args.folds)
    predicted = np.array([classes[i] for i in probabilities.argmax(axis=1)])
    confidence = probabilities.max(axis=1)

    print(f"\nsplit by visit, {args.folds} folds, no augmentation")
    print(f"raw accuracy (no floor): {(predicted == y).mean():.1%}")

    print("\n  floor   accepted   precision   wrong cat fed   junk let in")
    for threshold in (0.5, 0.6, 0.7, 0.75, 0.8, floor, 0.9, 0.95):
        o = outcomes(y, predicted, confidence, threshold)
        accepted = o["accepted"] or 1
        correct = (predicted == y) & (confidence >= threshold)
        mark = "  <-- config" if abs(threshold - floor) < 1e-9 else ""
        print(f"  {threshold:.2f}    {o['accepted'] / o['total']:6.1%}      "
              f"{correct.sum() / accepted:6.1%}        {o['wrong_cat_fed']:5d}"
              f"         {o['junk_let_in']:5d}{mark}")

    o = outcomes(y, predicted, confidence, floor)
    print(f"\nat {floor:.2f}, over {o['n_cats']} photos of a cat and {o['n_junk']} of junk:")
    print(f"  right cat fed    {o['right_cat_fed']:5d}  ({o['right_cat_fed'] / o['n_cats']:.1%} of cat photos)")
    print(f"  wrong cat fed    {o['wrong_cat_fed']:5d}  ({o['wrong_cat_fed'] / o['n_cats']:.2%})")
    print(f"  cat turned away  {o['cat_turned_away']:5d}  ({o['cat_turned_away'] / o['n_cats']:.1%})")
    print(f"  junk let in      {o['junk_let_in']:5d}  ({o['junk_let_in'] / o['n_junk']:.1%} of junk photos)")

    print(f"\nper cat at {floor:.2f}:")
    print("  cat    photos   recognised   turned away   called another cat")
    for cat in sorted(set(y)):
        mine = y == cat
        confident = mine & (confidence >= floor)
        right = int((confident & (predicted == cat)).sum())
        other_cat = int((confident & (predicted != cat) & (predicted != OTHER)).sum())
        print(f"  {cat:<6} {int(mine.sum()):6d}   {right / mine.sum():9.1%}   "
              f"{1 - right / mine.sum():11.1%}   {other_cat:18d}")

    print("\nA visit is many frames, and one confident frame opens the lid, so a cat")
    print("turned away in this table is a single frame - not a cat left hungry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
