"""Move very-sure visits already sitting in proposed/ into sure/<cat>.

The bowl does this itself as each visit ends (presort.very_sure), but photos
filed before that existed stay in proposed/. This re-judges them with the
current model, visit by visit, and moves only the visits it is very sure of.

    .venv/bin/python tools/resort_sure.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catbowl.config import load_config  # noqa: E402
from catbowl.embedder import build_embedder  # noqa: E402
from catbowl.presort import Shot, group_visits, taken_at, very_sure  # noqa: E402
from catbowl.recognizer import ClassifierBundle, Recognizer  # noqa: E402
from catbowl.sorting import PROPOSED, SURE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/bowls.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.capture.dir)
    bundle = ClassifierBundle.load(cfg.recognition.classifier)
    recognizer = Recognizer(build_embedder(cfg.recognition), bundle, cfg.recognition.min_confidence)

    paths = {}
    shots = []
    for photo in sorted((root / PROPOSED).glob("*/*.jpg")):
        image = cv2.imread(str(photo))
        if image is None:
            continue
        p = recognizer.predict(image)
        shots.append(Shot(name=photo.name, taken=taken_at(photo.name),
                          probabilities=dict(p.probabilities), label=p.raw_label,
                          confidence=p.confidence))
        paths[photo.name] = photo
    shots.sort(key=lambda s: (s.taken is None, s.taken or 0.0, s.name))
    group_visits(shots)

    moved = {}
    visits = {}
    for shot in shots:
        visits.setdefault(shot.visit, []).append(shot)
    for visit in visits.values():
        cat = very_sure(visit)
        if cat is None:
            continue
        target = root / SURE / cat
        for shot in visit:
            moved[cat] = moved.get(cat, 0) + 1
            if not args.dry_run:
                target.mkdir(parents=True, exist_ok=True)
                paths[shot.name].rename(target / shot.name)

    print(f"{len(shots)} photos in {len(visits)} visits; "
          f"{'would move' if args.dry_run else 'moved'} {sum(moved.values())} into sure/: {moved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
