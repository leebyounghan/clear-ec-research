"""Cell-route lane (2026-11-09): is the click scatter (markers vs machine consensus = mean of peaks and seg centroid) constant in pixels or in spacing?"""
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source')
import gate25 as G
def match(p, m, r):
    D = np.linalg.norm(p[:, None] - m[None], axis=2); D2 = np.where(D <= r, D, 1e6); a, c = linear_sum_assignment(D2); ok = D2[a, c] < 1e5; return a[ok], c[ok]
out = []
for i in range(25):
    mk = G.MK[i]; s = G.SPM[i]; hull = Delaunay(mk); pk = G.PEAKS['seeds3_tta4'][G.DS[i]]; p = pk[pk[:, 2] >= 0.2, :2]; p = p[hull.find_simplex(p) >= 0]
    sg = G.SEG[G.DS[i]]; sg = sg[sg[:, 3] >= 0.15 * s ** 2, :2]; a, c = match(p, sg, 0.4 * s); cons = 0.5 * (p[a] + sg[c])
    a2, c2 = match(cons, mk, 0.5 * s); d = cons[a2] - mk[c2]; d = d - d.mean(0); sc = np.sqrt((d ** 2).sum(1)).mean()
    out.append([s, sc, sc / s, G.gt_cv[i], G.gt_hex[i]])
O = np.array(out)
print('spacing px median %.1f (range %.1f-%.1f); click scatter px median %.2f, /s median %.3f' % (np.median(O[:, 0]), O[:, 0].min(), O[:, 0].max(), np.median(O[:, 1]), np.median(O[:, 2])))
print('corr(scatter_px, s) %.3f | corr(scatter/s, s) %.3f | slope px per px %.3f' % (np.corrcoef(O[:, 1], O[:, 0])[0, 1], np.corrcoef(O[:, 2], O[:, 0])[0, 1], np.polyfit(O[:, 0], O[:, 1], 1)[0]))
print('corr(scatter/s, label CV) %.3f | corr(scatter/s, label HEX) %.3f' % (np.corrcoef(O[:, 2], O[:, 3])[0, 1], np.corrcoef(O[:, 2], O[:, 4])[0, 1]))
