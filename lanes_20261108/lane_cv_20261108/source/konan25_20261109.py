"""CV lane (2026-11-09): Konan centre-method gate (KG takahashi1996konanpatent). Cell polygons from expert markers with power-weighted bisectors:
on each Delaunay edge i-j the cut point divides the segment in the ratio d_i^k : d_j^k (d = the cell's mean neighbour distance; k=0 is Voronoi).
Interior cells only (not on the convex hull, max angular gap between neighbours < 100 deg). Polygon = box clipped by the half-planes of all neighbours.
CV = SD/mean area, HEX = share of 6-sided interior cells. Read on the 25 expert frames against device CV/HEX: r and LOO-scaled MAPE for k in {0,2,3,4},
and k chosen leave-one-out (min LOO-scaled MAPE on the other 24 frames)."""
import json, sys
import numpy as np
from scipy.spatial import Delaunay
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
ann = json.load(open(C.X.parent / 'datasets/clear_ec_official/green_eval/annotations.json')); names = list(ann)
gt_cv = np.array([ann[n]['gt']['CV'] for n in names]); gt_hex = np.array([ann[n]['gt']['HEX'] for n in names]); MK = [np.asarray(ann[n]['markers_mha'], float) for n in names]

def clip(poly, n, c):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        fa, fb = (a - c) @ n, (b - c) @ n
        if fa <= 0: out.append(a)
        if fa * fb < 0: out.append(a + (b - a) * (fa / (fa - fb)))
    return np.array(out) if out else np.zeros((0, 2))

def cells(p, k):
    tri = Delaunay(p); hull = set(np.unique(tri.convex_hull)); nb = [set() for _ in range(len(p))]
    for s in tri.simplices:
        for a in s:
            for b_ in s:
                if a != b_: nb[a].add(b_)
    d = np.array([np.mean([np.linalg.norm(p[j] - p[i]) for j in nb[i]]) for i in range(len(p))])
    areas, sides = [], []
    for i in range(len(p)):
        if i in hull or len(nb[i]) < 3: continue
        ang = np.sort([np.arctan2(*(p[j] - p[i])[::-1]) for j in nb[i]]); gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        if gaps.max() > np.deg2rad(100): continue
        R = 3 * d[i]; poly = p[i] + np.array([[-R, -R], [R, -R], [R, R], [-R, R]])
        for j in nb[i]:
            v = p[j] - p[i]; L = np.linalg.norm(v); n = v / L; t = d[i] ** k / (d[i] ** k + d[j] ** k); poly = clip(poly, n, p[i] + t * v)
            if len(poly) < 3: break
        if len(poly) < 3: continue
        x, y = poly[:, 0], poly[:, 1]; areas.append(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        e = np.linalg.norm(poly - np.roll(poly, -1, 0), axis=1); sides.append(int((e > 0.08 * d[i]).sum()))
    areas = np.array(areas); return areas.std() / areas.mean(), np.mean(np.array(sides) == 6), len(areas)

def loo_mape(x, g):
    lx, lg_ = np.log(x), np.log(g); out = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i; pr = np.exp(lx[i] - lx[m].mean() + lg_[m].mean()); out.append(abs(pr - g[i]) / g[i])
    return 100 * np.mean(out)

K = (0, 2, 3, 4); R_ = {k: np.array([cells(p, k) for p in MK]) for k in K}
for k in K:
    cv, hx, n = R_[k][:, 0], R_[k][:, 1], R_[k][:, 2]
    print('k=%d  CV r %.3f LOO-MAPE %.2f scale (device/est) %.2f | HEX r %.3f LOO-MAPE %.2f | cells %d' % (k, np.corrcoef(np.log(cv), np.log(gt_cv))[0, 1], loo_mape(cv, gt_cv), np.exp(np.mean(np.log(gt_cv) - np.log(cv))), np.corrcoef(hx, gt_hex)[0, 1], loo_mape(np.maximum(hx, 1e-3), gt_hex), np.median(n)), flush=True)
# k chosen leave-one-out on CV
errs = []; picks = []
for i in range(len(names)):
    m = np.arange(len(names)) != i; best = min((k for k in K if k > 0), key=lambda k: loo_mape(R_[k][m, 0], gt_cv[m])); picks.append(best)
    lx, lg_ = np.log(R_[best][:, 0]), np.log(gt_cv); pr = np.exp(lx[i] - lx[m].mean() + lg_[m].mean()); errs.append(abs(pr - gt_cv[i]) / gt_cv[i])
print('k chosen leave-one-out (2-4): picks', dict(zip(*np.unique(picks, return_counts=True))), 'CV LOO-MAPE %.2f (marker Voronoi k=0: %.2f)' % (100 * np.mean(errs), loo_mape(R_[0][:, 0], gt_cv)), flush=True)
