"""Cell-route lane (2026-11-09): is detector jitter a per-frame offset or random scatter? Matched pairs (12 px, Hungarian) in the marker hull."""
import sys, time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source')
import gate25 as G
import cr_lib as L
def pairs(p, m, r=12.0):
    D = np.linalg.norm(p[:, None] - m[None], axis=2); D2 = np.where(D <= r, D, 1e6); a, c = linear_sum_assignment(D2); ok = D2[a, c] < 1e5; return p[a[ok]], m[c[ok]]
for nm, v in (('seeds3 t0.2', dict(src='seeds3_tta4', t=0.2)), ('s46 t0.2', dict(src='s46_tta4', t=0.2)), ('seg', dict(src='seg')), ('vote3+seg min1', dict(src='vote', members=['s46_tta4', 's47_tta4', 's48_tta4', 'seg'], minv=1))):
    off, res, tot = [], [], []
    for i in range(25):
        p, w, _, _ = G.points(i, v); hull = Delaunay(G.MK[i]); p = p[hull.find_simplex(p) >= 0]; a, b = pairs(p, G.MK[i]); dlt = a - b; s = G.SPM[i]
        mo = dlt.mean(0); off.append(np.linalg.norm(mo) / s); res.append(np.sqrt(((dlt - mo) ** 2).sum(1)).mean() / s); tot.append(np.sqrt((dlt ** 2).sum(1)).mean() / s)
    print(nm.ljust(16), 'mean |d|/s %.3f | per-frame offset |mean d|/s %.3f | residual after offset %.3f' % (np.mean(tot), np.mean(off), np.mean(res)), flush=True)
# timing for all-frame use: one frame, all t0.2 peaks
i = 0; pk = G.PEAKS['seeds3_tta4'][G.DS[i]]; q = pk[pk[:, 2] >= 0.2]; t = time.time(); res = L.konan_cells(q[:, :2]); t1 = time.time() - t; t = time.time(); pc, wc, a, b = L.clean(q[:, :2], q[:, 2]); t2 = time.time() - t
print('frame 0: %d peaks, konan_cells %.2f s, clean %.2f s (+%d virtual)' % (len(q), t1, t2, b))
