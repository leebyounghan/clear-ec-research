"""Measures the pre-only detector's errors on the 25 expert sets for the simulator's noise model (2026-11-05), per frame and pooled, peaks >= 0.2 inside the marker hull dilated by half a spacing: confidence quantiles of matched and false peaks, miss, FP, jitter at 0.35 and 0.5; offsets of false peaks in spacing units from the nearest marker; matched peak jitter vector SD. Also real graph statistics to match: sides histogram, node counts per frame, log area SD, confidence quantiles over all frames."""
import json, sys
import numpy as np
from scipy.spatial import Delaunay, cKDTree
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import abs_tess_20261104 as T
ann = json.load(open(f'{T.ROOT}/datasets/clear_ec_official/green_eval/annotations.json'))
with np.load(f'{T.ROOT}/datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z: ids = z['ids'].astype(str)
pos = {n: i for i, n in enumerate(ids)}; P = T.load_peaks(); tp, fp, off = [], [], []; rows = []
for n, e in ann.items():
    gt = np.asarray(e['markers_mha'], float); sp = float(np.median(cKDTree(gt).query(gt, k=2)[0][:, 1])); pk = P[pos[n]]; hull = Delaunay(gt); d0, _ = cKDTree(gt).query(pk[:, :2]); q = pk[(hull.find_simplex(pk[:, :2]) >= 0) | (d0 <= 0.5 * sp)]; q = q[np.argsort(-q[:, 2])]; used = set(); m = np.zeros(len(q), bool); dd = np.zeros(len(q))
    tree = cKDTree(gt)
    for k, (x, y, c) in enumerate(q):
        for dist, j in zip(*tree.query([x, y], k=3)):
            if dist <= 0.5 * sp and j not in used:
                used.add(j); m[k] = True; dd[k] = dist / sp; break
        if not m[k]: dd[k] = tree.query([x, y])[0] / sp
    tp += q[m, 2].tolist(); fp += q[~m, 2].tolist(); off += dd[~m].tolist()
    for t in (0.35, 0.5):
        s = q[:, 2] >= t; used_n = int((m & s).sum()); rows.append((n, t, 1 - used_n / len(gt), int((~m & s).sum()) / len(gt), float(np.median(dd[m & s])) if used_n else np.nan))
tp, fp, off = np.array(tp), np.array(fp), np.array(off); Q = (5, 10, 25, 50, 75, 90, 95)
print('matched conf quantiles', np.round(np.percentile(tp, Q), 3).tolist(), 'n', len(tp)); print('false conf quantiles', np.round(np.percentile(fp, Q), 3).tolist(), 'n', len(fp)); print('false offset from nearest marker, x spacing', np.round(np.percentile(off, Q), 3).tolist())
for t in (0.35, 0.5):
    v = np.array([r[2:] for r in rows if r[1] == t], float); print('t', t, 'miss p10 50 90', np.round(np.nanpercentile(v[:, 0], (10, 50, 90)), 3).tolist(), 'fp', np.round(np.nanpercentile(v[:, 1], (10, 50, 90)), 3).tolist(), 'jitter', np.round(np.nanpercentile(v[:, 2], (10, 50, 90)), 3).tolist())
z = np.load(f'{T.ROOT}/experiments/meta_ceiling_20261103/sweep/cell_graphs.npz'); X = z['X']; print('real nodes', X.shape, 'conf quantiles', np.round(np.percentile(X[:, 0], Q), 3).tolist(), 'sides share 3..9', np.round(np.bincount(X[:, 4].astype(int), minlength=10)[3:10] / len(X), 3).tolist(), 'unbounded', round(float(X[:, 6].mean()), 3), 'log area SD', round(float(X[:, 1].std()), 3), 'log dist SD', round(float(X[:, 2].std()), 3)); n = np.diff(z['offs']); print('nodes per frame p5 50 95', np.percentile(n[n > 0], (5, 50, 95)).tolist())
