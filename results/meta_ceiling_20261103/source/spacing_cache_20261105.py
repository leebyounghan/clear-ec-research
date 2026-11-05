"""Frame-indexed label-free cell spacing and canonical zoom (2026-11-05). d = median NN distance of pre-only peaks >= 0.35; frames with fewer than 30 peaks use D0. D0 = median d over all frames. s = D0 / d, clipped to [0.7, 1.45]. Writes sweep/spacing_t35.npz."""
import sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import abs_tess_20261104 as T
PK = T.load_peaks(); d = np.full(8954, np.nan)
for r, p in PK.items():
    q = p[p[:, 2] >= 0.35, :2]
    if len(q) >= 30: d[r] = np.median(cKDTree(q).query(q, k=2)[0][:, 1])
d0 = float(np.nanmedian(d)); s = np.where(np.isfinite(d), d0 / d, 1.0); s = np.clip(s, 0.7, 1.45)
np.savez('/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/spacing_t35.npz', d=d, scale=s, d0=d0)
print('frames', len(PK), 'with spacing', int(np.isfinite(d).sum()), 'd0', round(d0, 2), 'd p1 p5 p50 p95 p99', np.round(np.nanpercentile(d, (1, 5, 50, 95, 99)), 1).tolist(), 'clipped low', int((s <= 0.7).sum()), 'high', int((s >= 1.45).sum()))
