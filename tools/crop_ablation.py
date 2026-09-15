"""Do the old tight-crop photos help or hurt? Trains on all / new crops only /
new crops + all K, and scores each on visits since the crop fix reached the Pi
(2026-09-12 21:00). Written 2026-09-14; see docs/plan.md.

Usage: .venv/bin/python tools/crop_ablation.py
"""
import re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from threshold_report import visits_of, embed
from catbowl.config import load_config
from catbowl.training import load_dataset
from catbowl.recognizer import OTHER
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

CUTOFF = "20260912210000"
cfg = load_config("config/bowls.yaml")
ds = load_dataset("data/collected", ["J", "K", "F"], ["discard", "M"])
y = np.array(ds.labels); groups = visits_of(ds.paths)
X = embed(ds, cfg.recognition, False)
stamp = np.array([("".join(re.search(r"-(20\d{6})-(\d{6})", p.name).groups())) for p in ds.paths])
new = stamp >= CUTOFF
classes = sorted(set(y))
print("new-crop photos:", dict(zip(*np.unique(y[new], return_counts=True))))
print("old-crop photos:", dict(zip(*np.unique(y[~new], return_counts=True))))
print("new-crop visits:", {c: len(set(groups[new & (y == c)])) for c in classes})

def fit_predict(train_idx, test_idx):
    m = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced").fit(X[train_idx], y[train_idx])
    P = np.zeros((len(test_idx), len(classes)))
    col = {l: i for i, l in enumerate(m.classes_)}
    for j, l in enumerate(classes):
        if l in col: P[:, j] = m.predict_proba(X[test_idx])[:, col[l]]
    return P

new_idx = np.where(new)[0]
schemes = {"all photos (what runs now)": lambda tr: tr,
           "new crops only":             lambda tr: tr[new[tr]],
           "new crops + all K":          lambda tr: tr[new[tr] | (y[tr] == "K")]}
probs = {k: np.zeros((len(y), len(classes))) for k in schemes}
for _, te_local in GroupKFold(n_splits=5).split(new_idx, y[new_idx], groups[new_idx]):
    te = new_idx[te_local]
    tr = np.where(~np.isin(groups, groups[te]))[0]          # nothing from a test visit
    for k, pick in schemes.items():
        probs[k][te] = fit_predict(pick(tr), te)

def report(name, P, floor):
    P = P[new]; yt = y[new]; gt = groups[new]
    pred = np.array(classes)[P.argmax(1)]; conf = P.max(1)
    acc_mask = conf >= floor
    print(f"\n{name}   (floor {floor})")
    print(f"  raw accuracy {np.mean(pred == yt):.1%}   accepted {acc_mask.mean():.1%}   "
          f"precision {np.mean(pred[acc_mask] == yt[acc_mask]):.1%}")
    print("  per visit   opens for it   wrong cat   never opens")
    for c in classes:
        ids = sorted(set(gt[yt == c])); o = w = s = 0
        for v in ids:
            names = set(pred[(gt == v) & (yt == c) & acc_mask])
            if c == OTHER:
                o += bool(names - {OTHER}); continue
            o += c in names; w += bool(names - {c, OTHER}); s += not (names - {OTHER})
        if c == OTHER:
            print(f"  junk {len(ids):4d}    {o/len(ids):6.1%} would open a lid")
        else:
            print(f"  {c:<4} {len(ids):4d}    {o/len(ids):9.1%}    {w/len(ids):8.1%}   {s/len(ids):8.1%}")

for floor in (0.85, 0.75):
    for k in schemes:
        report(k, probs[k], floor)
