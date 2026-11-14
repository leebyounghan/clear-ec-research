"""CV lane (2026-11-09): Konan weighted-bisector CV/HEX (k=2, and Voronoi k=0 for reference) from detector peaks on all frames.
Peaks: 3-seed x 4-flip averaged pre-only peaks (out/tta_peaks/seeds3_tta4), nodes >= T (arg), the 3 legible 380x300 windows (from peaks >= 0.5).
Per window: interior cells (not on the window's convex hull, neighbour gap < 100 deg); areas pooled over the 3 windows. Writes out/konan_feats_<T>.npz with
per-frame (dataset index order, 8954) cv_k0, cv_k2, hex_k0, hex_k2, n_cells. 4 workers. usage: konan_all_20261109.py T"""
import os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Delaunay
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import legible_window_features_20261105 as L
T = float(sys.argv[1]); PK = {}
for f in range(5):
    z = np.load(f'/workspace/clear-ec-research/experiments/lane_cv_20261108/out/tta_peaks/seeds3_tta4/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
    for k_, i in enumerate(ix): PK[int(i)] = pk[of[k_]:of[k_ + 1]]

def clip(poly, n, c):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        fa, fb = (a - c) @ n, (b - c) @ n
        if fa <= 0: out.append(a)
        if fa * fb < 0: out.append(a + (b - a) * (fa / (fa - fb)))
    return np.array(out) if out else np.zeros((0, 2))

def cells(p, ks=(0, 2)):
    res = {k: ([], []) for k in ks}
    if len(p) < 12: return res
    try: tri = Delaunay(p)
    except Exception: return res
    hull = set(np.unique(tri.convex_hull)); nb = [set() for _ in range(len(p))]
    for s in tri.simplices:
        for a in s:
            for b_ in s:
                if a != b_: nb[a].add(b_)
    d = np.array([np.mean([np.linalg.norm(p[j] - p[i]) for j in nb[i]]) if nb[i] else np.nan for i in range(len(p))])
    for i in range(len(p)):
        if i in hull or len(nb[i]) < 3: continue
        ang = np.sort([np.arctan2(*(p[j] - p[i])[::-1]) for j in nb[i]]); gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        if gaps.max() > np.deg2rad(100): continue
        for k in ks:
            R = 3 * d[i]; poly = p[i] + np.array([[-R, -R], [R, -R], [R, R], [-R, R]])
            for j in nb[i]:
                v = p[j] - p[i]; n = v / np.linalg.norm(v); t = d[i] ** k / (d[i] ** k + d[j] ** k); poly = clip(poly, n, p[i] + t * v)
                if len(poly) < 3: break
            if len(poly) < 3: continue
            x, y = poly[:, 0], poly[:, 1]; res[k][0].append(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
            e = np.linalg.norm(poly - np.roll(poly, -1, 0), axis=1); res[k][1].append(int((e > 0.08 * d[i]).sum()))
    return res

def one(r):
    pk = PK.get(r); out = [np.nan] * 5
    if pk is None: return r, out
    hi = pk[pk[:, 2] >= 0.5, :2].astype(float); q = pk[pk[:, 2] >= T, :2].astype(float)
    if len(hi) < 10: return r, out
    A = {0: ([], []), 2: ([], [])}
    for x0, y0 in L.windows(hi, 380, 300, k=3):
        m = (q[:, 0] >= x0) & (q[:, 0] < x0 + 380) & (q[:, 1] >= y0) & (q[:, 1] < y0 + 300); res = cells(q[m])
        for k in (0, 2): A[k][0].extend(res[k][0]); A[k][1].extend(res[k][1])
    if len(A[2][0]) < 10: return r, out
    a0, a2 = np.array(A[0][0]), np.array(A[2][0])
    return r, [a0.std() / a0.mean(), a2.std() / a2.mean(), np.mean(np.array(A[0][1]) == 6), np.mean(np.array(A[2][1]) == 6), len(a2)]

if __name__ == '__main__':
    F = np.full((8954, 5), np.nan)
    with get_context('fork').Pool(4) as pool:
        for r, v in pool.imap_unordered(one, range(8954), chunksize=16): F[r] = v
    np.savez_compressed(f'/workspace/clear-ec-research/experiments/lane_cv_20261108/out/konan_feats_t{T}.npz', F=F, names=np.array(['cv_k0', 'cv_k2', 'hex_k0', 'hex_k2', 'n_cells']), note='dataset index order; peaks seeds3_tta4; 3 legible windows')
    print('done', int(np.isfinite(F[:, 1]).sum()), 'frames', flush=True)
