"""Retrain on the sorted Pi-camera photos, if they have changed, and score it.

Run every night by `catbowl-train.timer`. It:

1. gathers the sorted photos taken since the ribbon camera went in (CUTOFF) for
   J, K, F and discard. M and unclear are left out (decided 2026-09-24);
2. stops if that set is the same as the one the current model was trained on;
3. scores it per visit the same way `tools/threshold_report.py` does, at the
   config's min_confidence;
4. trains the classifier on all of it, backs up the old one to models/backups/
   and swaps the new one in;
5. appends the scores to models/history.json (shown on the status page) and
   leaves models/.restart for the unit to restart catbowl.

    .venv/bin/python tools/nightly_train.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_report import OTHER, embed, load_config, load_dataset, out_of_fold, visits_of  # noqa: E402

CUTOFF = "20260922-0948"          # first photo from the ribbon camera
LABELS = ["J", "K", "F"]
NEGATIVE = ["discard"]


def taken_at(name: str) -> str:
    """'bowl1-20260922-101500-123.jpg' -> '20260922-1015'."""
    parts = name.split("-")
    return f"{parts[1]}-{parts[2][:4]}" if len(parts) >= 3 else ""


def gather(collected: Path, subset: Path) -> list[str]:
    """Rebuild *subset* as symlinks to the Pi-camera photos; return their names."""
    shutil.rmtree(subset, ignore_errors=True)
    names = []
    for folder in LABELS + NEGATIVE:
        (subset / folder).mkdir(parents=True)
        for photo in sorted((collected / folder).glob("*.jpg")):
            if taken_at(photo.name) >= CUTOFF:
                (subset / folder / photo.name).symlink_to(photo.resolve())
                names.append(f"{folder}/{photo.name}")
    return names


def score(subset: Path, cfg) -> dict:
    """Per-visit outcomes at the config's floor, visits held out 5-fold."""
    dataset = load_dataset(str(subset), LABELS, NEGATIVE)
    y = np.array(dataset.labels)
    groups = visits_of(dataset.paths)
    X = embed(dataset, cfg.recognition, refresh=False)
    classes, probabilities = out_of_fold(X, y, groups, 5)
    predicted = np.array([classes[i] for i in probabilities.argmax(axis=1)])
    confidence = probabilities.max(axis=1)
    floor = cfg.recognition.min_confidence

    cats = {}
    for cat in LABELS:
        ids = sorted(set(groups[y == cat]))
        opened = wrong = 0
        for visit in ids:
            names = set(predicted[(groups == visit) & (y == cat) & (confidence >= floor)])
            opened += cat in names
            wrong += bool(names - {cat, OTHER})
        cats[cat] = {"photos": int((y == cat).sum()), "visits": len(ids),
                     "opens": round(opened / max(1, len(ids)), 3),
                     "wrong_cat": round(wrong / max(1, len(ids)), 3)}
    junk_ids = sorted(set(groups[y == OTHER]))
    junk_open = sum(bool(set(predicted[(groups == v) & (y == OTHER) & (confidence >= floor)]) - {OTHER})
                    for v in junk_ids)
    return {"floor": floor, "accuracy": round(float((predicted == y).mean()), 3), "cats": cats,
            "junk": {"photos": int((y == OTHER).sum()), "visits": len(junk_ids),
                     "opens": round(junk_open / max(1, len(junk_ids)), 3)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/bowls.yaml")
    parser.add_argument("--data", default="data/collected")
    parser.add_argument("--force", action="store_true", help="retrain even if nothing changed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = Path(cfg.recognition.classifier)
    history_path = model.parent / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    subset = Path("data/train-picam")
    names = gather(Path(args.data), subset)
    fingerprint = hashlib.sha1("\n".join(names).encode()).hexdigest()
    if history and history[-1].get("fingerprint") == fingerprint and not args.force:
        print("no new sorted photos since the last model; nothing to do")
        return 0

    started = time.time()
    stats = score(subset, cfg)
    new = model.with_suffix(".new.joblib")
    subprocess.run([sys.executable, "-m", "catbowl", "--config", args.config, "train",
                    "--data", str(subset), "--labels", *LABELS, "--negative", *NEGATIVE,
                    "--out", str(new)], check=True)

    backups = model.parent / "backups"
    backups.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    if model.exists():
        shutil.copy2(model, backups / f"classifier-{stamp}.joblib")
    os.replace(new, model)

    history.append({"trained": time.strftime("%Y-%m-%d %H:%M"), "fingerprint": fingerprint,
                    "since": CUTOFF, "minutes": round((time.time() - started) / 60, 1), **stats})
    history_path.write_text(json.dumps(history, indent=1))
    (model.parent / ".restart").touch()
    print(json.dumps(history[-1], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
