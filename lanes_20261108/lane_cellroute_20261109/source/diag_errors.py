"""Cell-route lane (2026-11-09): structure of real detection errors on the 25 marker frames (for a realistic corruption simulator).
Misses: share whose marker-Delaunay neighbours are also missed vs the share expected at random. FPs: distance to the nearest marker / spacing,
and to the nearest matched detection / spacing."""
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, cKDTree
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source')
import gate25 as G
import cr_lib as L
for nm, v in (('seeds3 t0.2', dict(src='seeds3_tta4', t=0.2)), ('seeds3 t0.35', dict(src='seeds3_tta4', t=0.35)), ('vote3+seg min1', dict(src='vote', members=['s46_tta4', 's47_tta4', 's48_tta4', 'seg'], minv=1))):
    nbm, exp_, fpd, fpdd, fpconf, tpconf = [], [], [], [], [], []
    for i in range(25):
        mk = G.MK[i]; s = G.SPM[i]; p, w, _, _ = G.points(i, v); hull = Delaunay(mk); ins = hull.find_simplex(p) >= 0; p, w = p[ins], w[ins]
        D = np.linalg.norm(p[:, None] - mk[None], axis=2); D2 = np.where(D <= 12, D, 1e6); a, c = linear_sum_assignment(D2); ok = D2[a, c] < 1e5
        hit = np.zeros(len(mk), bool); hit[c[ok]] = True; tp = np.zeros(len(p), bool); tp[a[ok]] = True
        nb = L.neighbours(Delaunay(mk)); miss = ~hit; rate = miss.mean()
        for j in np.where(miss)[0]: nbm.append(miss[nb[j]].mean())
        exp_.append(rate)
        if (~tp).any():
            fpd.extend(list(cKDTree(mk).query(p[~tp])[0] / s))
            if tp.any(): fpdd.extend(list(cKDTree(p[tp]).query(p[~tp])[0] / s))
            fpconf.extend(list(w[~tp]))
        tpconf.extend(list(w[tp]))
    q = lambda x: np.round(np.percentile(x, (10, 25, 50, 75, 90)), 2)
    print(nm.ljust(15), 'missed: share of missed neighbours %.3f vs miss rate %.3f (clustering x%.2f)' % (np.mean(nbm), np.mean(exp_), np.mean(nbm) / np.mean(exp_)), flush=True)
    print(' ' * 15, 'FP dist to nearest marker / s, pct 10-90:', q(fpd), '| to nearest matched detection / s:', q(fpdd), '| conf FP median %.2f vs TP %.2f' % (np.median(fpconf), np.median(tpconf)), flush=True)
