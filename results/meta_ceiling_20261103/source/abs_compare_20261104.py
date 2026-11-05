"""Fair comparison on the finished rows (2026-11-04): watershed topology vs Voronoi topology from the same single pre-only detector at the same threshold, vs the shipped
ten-detector hex6 mean. Pearson with log y and with B's log residual; added value of tessellation features over hex6 by OLS partial correlation."""
import json, os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Delaunay, Voronoi
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_oof_20261031 as E
import abs_tess_20261104 as T
K = list(E.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']
z = np.load(sys.argv[1]); names = [str(n) for n in z['names']]; F = z['F'][idx].astype(float); seen = z['seen'][idx]
hx = np.load('/workspace/clear-ec-research/experiments/green_track_20261028/deploy/hex6_blend_params_12_oof_feature.npz')['feature'][idx].astype(float)
ly = np.log(np.where(y > 0, y, np.nan)); r = ly - np.log(B); rows = np.where(seen)[0]; PK = T.load_peaks()
def vor(i):
    pk = PK[int(idx[i])]; out = []
    for t in (0.35, 0.5):
        q = pk[pk[:, 2] >= t, :2].astype(np.float64)
        if len(q) < 30: out += [np.nan, np.nan]; continue
        v = Voronoi(q); tri = Delaunay(q); si = []; ar = []
        for n in range(len(q)):
            g = v.regions[v.point_region[n]]
            if -1 in g or not g: continue
            w = v.vertices[g]
            if (tri.find_simplex(w) < 0).any(): continue
            si.append(len(g)); x_, y_ = w[:, 0], w[:, 1]; ar.append(0.5 * abs(np.dot(x_, np.roll(y_, 1)) - np.dot(y_, np.roll(x_, 1))))
        si = np.array(si); ar = np.array(ar); out += [float((si == 6).mean()) if len(si) > 11 else np.nan, float(ar.std() / ar.mean()) if len(si) > 11 else np.nan]
    return out
workers = int(min(4, max(2, 16 - os.getloadavg()[0] - 1)))
with get_context('fork').Pool(workers) as pool: V = np.array(pool.map(vor, rows.tolist(), chunksize=16))
def c(f, j):
    s = np.isfinite(f) & np.isfinite(r[rows, j]); return round(float(np.corrcoef(f[s], ly[rows[s], j])[0, 1]), 3), round(float(np.corrcoef(f[s], r[rows[s], j])[0, 1]), 3)
tab = {'hex6_ten': hx[rows], 'vor_hex_t35': np.log(V[:, 0]), 'vor_hex_t50': np.log(V[:, 2]), 'vor_cv_t35': np.log(V[:, 1]), 'vor_cv_t50': np.log(V[:, 3])}
for nm in ('t35_all_hex3', 't35_all_hex5', 't35_all_hex8', 't35s_all_hex5', 't50_all_hex5', 't50_all_hex3', 't35_all_cv', 't35_all_cv_trim', 't35s_all_cv_trim', 't50_all_cv_trim'):
    f = F[rows, names.index(nm)]; f[f <= 0] = np.nan; tab[nm] = np.log(f)
rep = {nm: {K[j]: c(v, j) for j in range(3)} for nm, v in tab.items()}
for nm, v in rep.items(): print(nm, json.dumps(v))
Xh = np.nan_to_num(hx[rows], nan=np.nanmedian(hx[rows]))
def partial(f, j):
    s = np.isfinite(f) & np.isfinite(r[rows, j]) & np.isfinite(Xh); A = np.c_[np.ones(s.sum()), Xh[s]]; res = lambda t: t - A @ np.linalg.lstsq(A, t, rcond=None)[0]
    return round(float(np.corrcoef(res(f[s]), res(r[rows[s], j]))[0, 1]), 3)
print('partial r with residual after removing hex6 (n=%d, SE ~ %.3f):' % (len(rows), 1 / np.sqrt(len(rows))))
for nm, v in tab.items():
    if nm != 'hex6_ten': print(' ', nm, {K[j]: partial(v, j) for j in range(3)})
print('done', flush=True)
