"""Cell-route lane (2026-11-09): (a) are 'misses + FPs' partly marker misregistration? Affine ICP of markers onto detections (pairs within 0.5 s),
then re-match at 12 px. (b) scatter between two machine point sets (detector peaks vs seg-mask centroids, seeds3 vs single seed) vs scatter to markers:
if machine-machine scatter << machine-marker scatter, the 0.19 s is mostly human click noise carried by the label."""
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, cKDTree
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source')
import gate25 as G
def match(p, m, r):
    D = np.linalg.norm(p[:, None] - m[None], axis=2); D2 = np.where(D <= r, D, 1e6); a, c = linear_sum_assignment(D2); ok = D2[a, c] < 1e5; return a[ok], c[ok]
def scatter(p, q, s, r=12.0):
    a, c = match(p, q, r); d = p[a] - q[c]; mo = d.mean(0); return np.sqrt(((d - mo) ** 2).sum(1)).mean() / s, len(a)
rows = []
for i in range(25):
    mk = G.MK[i]; s = G.SPM[i]; hull = Delaunay(mk)
    pk = G.PEAKS['seeds3_tta4'][G.DS[i]]; p = pk[pk[:, 2] >= 0.2, :2]; p = p[hull.find_simplex(p) >= 0]
    pk1 = G.PEAKS['s46_tta4'][G.DS[i]]; p1 = pk1[pk1[:, 2] >= 0.2, :2]; p1 = p1[hull.find_simplex(p1) >= 0]
    sg = G.SEG[G.DS[i]]; sg = sg[sg[:, 3] >= 0.15 * s ** 2, :2]; sg = sg[hull.find_simplex(sg) >= 0]
    a, c = match(p, mk, 12.0); miss0 = 1 - len(c) / len(mk); fp0 = 1 - len(a) / len(p)
    M = mk.copy()
    for it in range(6):
        tr = cKDTree(p); d, j = tr.query(M); ok = d < 0.5 * s
        X = np.c_[mk[ok], np.ones(ok.sum())]; A, *_ = np.linalg.lstsq(X, p[j[ok]], rcond=None); M = np.c_[mk, np.ones(len(mk))] @ A
    a, c = match(p, M, 12.0); miss1 = 1 - len(c) / len(mk); fp1 = 1 - len(a) / len(p); sc1 = np.sqrt(((p[a] - M[c]) ** 2).sum(1)).mean() / s
    lin = A[:2]; sv = np.linalg.svd(lin, compute_uv=False)
    rows.append([miss0, fp0, miss1, fp1, sc1, sv[0], sv[1], scatter(p, mk, s)[0], scatter(sg, mk, s)[0], scatter(p, sg, s)[0], scatter(p1, p, s)[0], scatter(p1, mk, s)[0]])
R = np.array(rows); m = R.mean(0)
print('seeds3 t0.2 vs markers: miss %.3f FP %.3f -> after affine ICP miss %.3f FP %.3f, scatter %.3f s; affine singular values median %.3f %.3f' % (m[0], m[1], m[2], m[3], m[4], np.median(R[:, 5]), np.median(R[:, 6])))
print('residual scatter / s (offset removed): peaks-markers %.3f | seg-markers %.3f | peaks-seg %.3f | s46-seeds3 %.3f | s46-markers %.3f' % tuple(m[7:]))
