"""Audit of beliefs never tested (2026-11-05).
  A1 scale: all frames carry the same pixel spacing in the header. CD should equal k / spacing^2 with one k. k_i = CD label x spacing_px^2, spacing from pre-only peaks: per source, per year; is log k unimodal; do fellow eyes share it beyond the label; does it explain CD error?
  A2 label formula: are the labels computed from the marked cells? On the 25 sets: ratio of label to several CD, CV, HEX estimators from the markers; rank correlations; which definition reaches r near 1.
  A3 label lattice: if ROI area or cell count is fixed, labels take discrete values and reveal N, and with it the sampling noise of HEX for every image, not just 25. Distinct values, repeats, spacing of sorted CD; HEX x N integer test.
  A4 hidden modes: is the residual a mixture? Kurtosis, joint distribution of residuals, clusters by source and year; do fellow eyes share a mode?"""
import json, re, sys
import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
from scipy import stats
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import abs_tess_20261104 as T
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; N = len(ids)
fmt = np.array([0 if re.fullmatch(r'\d{4}-\d{2}', i) else 1 for i in ids]); yr = np.array([2000 + int(i[-2:]) if f == 0 else int(i[:4]) for i, f in zip(ids, fmt)]); ly = np.log(np.where(y > 0, y, np.nan)); r = ly - np.log(B)
print('=== A3 label lattice')
for j, k in enumerate(K):
    for f, nm in ((0, 'old'), (1, 'new')):
        v = y[(fmt == f) & (y[:, j] > 0), j]; u, c = np.unique(v, return_counts=True); d = np.diff(u); print(k, nm, 'n', len(v), 'distinct', len(u), 'min', float(u.min()), 'max', float(u.max()), 'smallest gaps', np.unique(np.round(d, 4))[:6].tolist(), 'top repeats', [(float(u[i]), int(c[i])) for i in np.argsort(-c)[:6]])
for f, nm in ((0, 'old'), (1, 'new')):
    v = y[(fmt == f) & (y[:, 0] > 0), 0]; u = np.unique(v); print(nm, 'CD integer share', float(np.mean(v == np.round(v))), 'distinct per 100 CD units', round(len(u) / ((u.max() - u.min()) / 100), 2)); hist, ed = np.histogram(v, bins=np.arange(1500, 3600, 1)); s = np.abs(np.fft.rfft(hist - hist.mean())); fr = np.fft.rfftfreq(len(hist), 1.0); top = np.argsort(-s[5:])[:5] + 5; print('  periodicity in the CD histogram, strongest periods (CD units)', np.round(1 / fr[top], 2).tolist(), 'power vs median', np.round(s[top] / np.median(s[5:]), 1).tolist())
def nfit(h, cands):
    """Share of labels consistent with HEX = round(n6 / N, 2) for a given N."""
    return float(np.mean(np.abs(h * cands - np.round(h * cands)) <= cands * 0.005 + 1e-9))
print('=== A1 scale'); PK = T.load_peaks(); sp = np.full(N, np.nan)
for n in range(N):
    p = PK.get(int(idx[n]))
    if p is None: continue
    q = p[p[:, 2] >= 0.5, :2]
    if len(q) >= 30:
        d = cKDTree(q).query(q, k=2)[0][:, 1]; d = d[(d > 6)]; sp[n] = np.median(d)
lk = ly[:, 0] + 2 * np.log(sp); ok = np.isfinite(lk)
print('rows', int(ok.sum()), 'log k overall median', round(float(np.median(lk[ok])), 3), 'robust SD', round(float(1.4826 * np.median(np.abs(lk[ok] - np.median(lk[ok])))), 4), 'kurtosis', round(float(stats.kurtosis(lk[ok])), 2))
for f, nm in ((0, 'old'), (1, 'new')):
    s = ok & (fmt == f); print(nm, 'median log k', round(float(np.median(lk[s])), 4), 'robust SD', round(float(1.4826 * np.median(np.abs(lk[s] - np.median(lk[s])))), 4), 'corr with residual', round(float(np.corrcoef(lk[s & np.isfinite(r[:, 0])], r[s & np.isfinite(r[:, 0]), 0])[0, 1]), 3))
    print('  by year', {int(a): round(float(np.median(lk[s & (yr == a)])), 3) for a in np.unique(yr[s])})
    h, e = np.histogram(lk[s], bins=40, range=(np.median(lk[s]) - 0.6, np.median(lk[s]) + 0.6)); print('  histogram', h.tolist())
    print('  bimodality coefficient', round(float((stats.skew(lk[s]) ** 2 + 1) / (stats.kurtosis(lk[s]) + 3)), 3), '(uniform 0.555, above suggests more than one mode)')
new = {}
for n, i in enumerate(ids):
    m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
    if m: new[(m.group(1), m.group(2), m.group(3).replace('0', 'O'))] = n
P = np.array([(v, new[(k[0], k[1], 'OS')]) for k, v in new.items() if k[2] == 'OD' and (k[0], k[1], 'OS') in new]); s = np.isfinite(lk[P[:, 0]]) & np.isfinite(lk[P[:, 1]]); print('fellow eyes, correlation of log k', round(float(np.corrcoef(lk[P[s, 0]], lk[P[s, 1]])[0, 1]), 3), 'spacing alone', round(float(np.corrcoef(np.log(sp[P[s, 0]]), np.log(sp[P[s, 1]]))[0, 1]), 3), 'CD label', round(float(np.corrcoef(ly[P[s, 0], 0], ly[P[s, 1], 0])[0, 1]), 3))
print('log spacing vs log CD: slope (physics -0.5)', {nm: round(float(np.polyfit(ly[ok & (fmt == f), 0], np.log(sp[ok & (fmt == f)]), 1)[0]), 3) for f, nm in ((0, 'old'), (1, 'new'))}, 'pearson', {nm: round(float(np.corrcoef(ly[ok & (fmt == f), 0], np.log(sp[ok & (fmt == f)]))[0, 1]), 3) for f, nm in ((0, 'old'), (1, 'new'))})
print('=== A2 label formula on the 25 marker sets'); ann = json.load(open('/workspace/clear-ec-research/datasets/clear_ec_official/green_eval/annotations.json')); UM = 1000 / 1296; rows = []
def polys(pts):
    vor = Voronoi(pts); tri = Delaunay(pts); A, S, D = [], [], []
    nb = {i: set() for i in range(len(pts))}
    for a_, c_ in vor.ridge_points: nb[a_].add(c_); nb[c_].add(a_)
    for i in range(len(pts)):
        g = vor.regions[vor.point_region[i]]
        if -1 in g or not g: continue
        v = vor.vertices[g]
        if (tri.find_simplex(v) < 0).any(): continue
        x, z = v[:, 0], v[:, 1]; A.append(0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))) * UM * UM); S.append(len(g)); D.append(np.mean([np.linalg.norm(pts[i] - pts[q]) for q in nb[i]]) * UM)
    return np.array(A), np.array(S), np.array(D)
for n, e in ann.items():
    p = np.asarray(e['markers_mha'], float); A, S, D = polys(p); g = e['gt']; t = Delaunay(p); hull = sum(0.5 * abs(np.cross(p[s[1]] - p[s[0]], p[s[2]] - p[s[0]])) for s in t.simplices) * UM * UM; dnn = cKDTree(p).query(p, k=2)[0][:, 1] * UM
    x0, y0, x1, y1 = e['roi_box_mha']; box = (x1 - x0) * (y1 - y0) * UM * UM; tri_a = np.array([0.5 * abs(np.cross(p[s[1]] - p[s[0]], p[s[2]] - p[s[0]])) for s in t.simplices]) * UM * UM
    rows.append(dict(id=n, n=len(p), bounded=len(A), CD=g['CD'], CV=g['CV'], HEX=g['HEX'], cd_vor=1e6 / A.mean(), cd_hull=1e6 * len(p) / hull, cd_hull_adj=1e6 * (len(p) - 0.5 * len(t.convex_hull) - 1 + 0) / hull if False else 1e6 * (len(p) - 0.5 * len(t.convex_hull)) / hull, cd_box=1e6 * len(p) / box, cv_area=A.std() / A.mean(), cv_area_med=1.4826 * np.median(np.abs(A - np.median(A))) / np.median(A), cv_dist=D.std() / D.mean(), cv_nn=dnn.std() / dnn.mean(), cv_tri=tri_a.std() / tri_a.mean(), hex6=(S == 6).mean(), hex_all=None, area_box=box, area_hull=hull))
tab = {}
for t_, cols in (('CD', ['cd_vor', 'cd_hull', 'cd_hull_adj', 'cd_box']), ('CV', ['cv_area', 'cv_area_med', 'cv_dist', 'cv_nn', 'cv_tri']), ('HEX', ['hex6'])):
    gt_ = np.array([q[t_] for q in rows])
    for c in cols:
        v = np.array([q[c] for q in rows]); tab[c] = dict(r=round(float(np.corrcoef(v, gt_)[0, 1]), 3), spearman=round(float(stats.spearmanr(v, gt_)[0]), 3), median_ratio=round(float(np.median(gt_ / v)), 3), loo_mape=round(float(np.mean([abs(v[i] * np.median(np.delete(gt_ / v, i)) - gt_[i]) / gt_[i] for i in range(len(v))]) * 100), 2))
    print(t_, json.dumps({c: tab[c] for c in cols}))
print('per image: id, n markers, area of box vs hull (mm2), label CD, cd_hull_adj, ratio, label CV, cv_area, ratio, label HEX, hex6')
for q in rows: print(' ', q['id'], q['n'], round(q['area_box'] / 1e6, 4), round(q['area_hull'] / 1e6, 4), q['CD'], round(q['cd_hull_adj']), round(q['CD'] / q['cd_hull_adj'], 3), q['CV'], round(q['cv_area'], 3), round(q['CV'] / q['cv_area'], 2), q['HEX'], round(q['hex6'], 3))
print('=== A4 hidden modes')
for j, k in enumerate(K):
    for f, nm in ((0, 'old'), (1, 'new')):
        q = (fmt == f) & np.isfinite(r[:, j]); v = r[q, j]; print(k, nm, 'kurtosis', round(float(stats.kurtosis(v)), 2), 'skew', round(float(stats.skew(v)), 2), 'bimodality', round(float((stats.skew(v) ** 2 + 1) / (stats.kurtosis(v) + 3)), 3), 'quantiles 1,10,50,90,99', np.round(np.quantile(v, [.01, .1, .5, .9, .99]), 3).tolist())
q = np.isfinite(r).all(1); print('residual correlations CD-CV, CD-HEX, CV-HEX', {nm: [round(float(np.corrcoef(r[q & (fmt == f), a], r[q & (fmt == f), c])[0, 1]), 3) for a, c in ((0, 1), (0, 2), (1, 2))] for f, nm in ((0, 'old'), (1, 'new'))}, 'label correlations', {nm: [round(float(np.corrcoef(ly[q & (fmt == f), a], ly[q & (fmt == f), c])[0, 1]), 3) for a, c in ((0, 1), (0, 2), (1, 2))] for f, nm in ((0, 'old'), (1, 'new'))})
print('done', flush=True)
