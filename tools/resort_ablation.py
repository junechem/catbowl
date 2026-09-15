"""Does the re-sorted discard pile help? Every way of training discard, M and unclear,
on the same visit-grouped folds. Written 2026-09-14; see docs/plan.md.

Usage: .venv/bin/python tools/resort_ablation.py
"""
import re, sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, ".")
from catbowl.config import load_config
from catbowl.embedder import build_embedder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path("data/collected"); CUTOFF = "20260912210000"; GAP = 20.0; FLOOR = 0.85
FOLDERS = ["F", "J", "K", "discard", "M", "unclear"]
CATS = ["F", "J", "K"]
CACHE = Path("~/.cache/catbowl-embeddings.npz").expanduser()
CACHE2 = Path("~/.cache/catbowl-embeddings-byname.npz").expanduser()

paths, folder = [], []
for d in FOLDERS:
    for p in sorted((ROOT / d).glob("*.jpg")):
        paths.append(p); folder.append(d)
folder = np.array(folder)
names = np.array([p.name for p in paths])

# embeddings, reused by file name (moving a photo does not change it)
known = {}
for c in (CACHE, CACHE2):
    if c.exists():
        z = np.load(c, allow_pickle=True)
        for n, v in zip(z["names"], z["X"]): known[Path(str(n)).name] = v
missing = [i for i, n in enumerate(names) if n not in known]
if missing:
    emb = build_embedder(load_config("config/bowls.yaml").recognition)
    print(f"embedding {len(missing)} new photos")
    for s in range(0, len(missing), 16):
        idx = missing[s:s + 16]
        for i, v in zip(idx, emb.embed_batch([cv2.imread(str(paths[i])) for i in idx])):
            known[names[i]] = v
X = np.stack([known[n] for n in names])
np.savez_compressed(CACHE2, X=X, names=names)

# visits by time alone, whatever folder a frame was filed in
def ts(n):
    m = re.search(r"-(20\d{6})-(\d{6})-(\d{3})", n)
    from datetime import datetime
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp() + int(m.group(3)) / 1000
t = np.array([ts(n) for n in names]); order = np.argsort(t)
groups = np.zeros(len(t), int); g = 0; prev = None
for i in order:
    if prev is None or t[i] - prev > GAP: g += 1
    groups[i] = g; prev = t[i]
new = np.array([re.search(r"-(20\d{6})-(\d{6})", n).group(0)[1:].replace("-", "") >= CUTOFF for n in names])

counts = {d: int((folder == d).sum()) for d in FOLDERS}
print("photos:", counts, " visits:", len(set(groups)))

schemes = {
    "before (discard+M+unclear = not a cat)": {"discard": "_other", "M": "_other", "unclear": "_other"},
    "A: discard only, M & unclear left out":  {"discard": "_other"},
    "B: discard+M = not a cat, unclear out":  {"discard": "_other", "M": "_other"},
    "C: discard = not a cat, M = crowd":      {"discard": "_other", "M": "_crowd"},
}
def target(scheme):
    y = np.array([d if d in CATS else scheme.get(d, "") for d in folder])
    return y  # "" = not trained

folds = list(GroupKFold(n_splits=5).split(X, folder, groups))
results = {}
for name, scheme in schemes.items():
    y = target(scheme); classes = sorted(set(y) - {""})
    P = np.zeros((len(y), len(classes)))
    for tr, te in folds:
        tr = tr[y[tr] != ""]
        m = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
        col = {l: i for i, l in enumerate(m.classes_)}
        pp = m.predict_proba(X[te])
        for j, l in enumerate(classes):
            if l in col: P[te, j] = pp[:, col[l]]
    results[name] = (classes, P)

def table(mask, title):
    print(f"\n==== {title}  (per visit, floor {FLOOR}) ====")
    hdr = f"{'':42s}" + "".join(f"{c+' opens':>10s}{'wrong':>7s}" for c in CATS) + \
          f"{'junk':>7s}{'M open':>8s}{'M crowd':>8s}{'uncl.':>7s}"
    print(hdr)
    for name, (classes, P) in results.items():
        pred = np.array(classes)[P.argmax(1)]; conf = P.max(1); sure = conf >= FLOOR
        row = f"{name:42s}"
        def visits(d): return sorted(set(groups[mask & (folder == d)]))
        def called(v, d): return set(pred[(groups == v) & (folder == d) & mask & sure])
        for c in CATS:
            ids = visits(c)
            o = sum(c in called(v, c) for v in ids) / len(ids)
            w = sum(bool(called(v, c) & (set(CATS) - {c})) for v in ids) / len(ids)
            row += f"{o:10.1%}{w:7.1%}"
        opens = lambda d: sum(bool(called(v, d) & set(CATS)) for v in visits(d)) / max(1, len(visits(d)))
        crowd = sum("_crowd" in called(v, "M") for v in visits("M")) / max(1, len(visits("M")))
        row += f"{opens('discard'):7.1%}{opens('M'):8.1%}{(crowd if '_crowd' in classes else float('nan')):8.1%}{opens('unclear'):7.1%}"
        print(row)
    print("visits: " + ", ".join(f"{d} {len(set(groups[mask & (folder == d)]))}" for d in FOLDERS))

table(np.ones(len(names), bool), "all photos")
table(new, "new crops only (since 2026-09-12 21:00)")
print("\njunk / M open = share of those visits where a confident cat frame would open the lid (lower is better)")
print("M crowd = share of multi-cat visits the classifier itself flags as a crowd")
print("uncl. = share of unclear-cat visits that open for some cat (not wrong in itself: it is F or J)")

# C only: how often does a single cat get called a crowd? 4 of any 6 consecutive
# confident crowd frames in one visit is what closes an open lid.
classes, P = results["C: discard = not a cat, M = crowd"]
pred = np.array(classes)[P.argmax(1)]; sure = P.max(1) >= FLOOR
print("\nC: single-cat visits with a crowd call")
for c in CATS:
    ids = sorted(set(groups[folder == c])); anyc = closes = 0
    for v in ids:
        idx = np.where((groups == v) & (folder == c))[0]; idx = idx[np.argsort(t[idx])]
        hit = (pred[idx] == "_crowd") & sure[idx]
        anyc += hit.any()
        closes += any(hit[k:k + 6].sum() >= 4 for k in range(len(hit)))
    frames = ((pred == "_crowd") & sure & (folder == c)).sum() / (folder == c).sum()
    print(f"  {c}: {anyc/len(ids):5.1%} of visits have one, {closes/len(ids):5.1%} would close the lid (4 of 6); {frames:.1%} of frames")
