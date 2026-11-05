"""Cell statistics inside the most legible window, all frames (2026-11-05). On the 25 expert sets the window with the most confident pre-only peaks overlaps the expert region at IoU 0.57-0.61 (random 0.11-0.24). Label-free, so the 8953 labels are a fair test.
For each frame and each window size (hull-like 380 x 300, box 538 x 408) the window with the most peaks >= 0.5 (stride 16) is selected; Voronoi on peaks >= 0.35 within it: hex6, area CV, CV of neighbour distance, CD from mean area, cell count. Same statistics on the whole frame and on the centre window as controls. Also the top three non-overlapping windows averaged. Forked pool, clamp(16 - load1 - 1, 2, 12). Writes sweep/legible_window.npz."""
import json, os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Delaunay, Voronoi
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import abs_tess_20261104 as T
H, W = T.H, T.W; UM = 1000 / 1296; OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/legible_window.npz'
SIZES = dict(hull=(380, 300), box=(538, 408)); STATS = ['hex6', 'cv_area', 'cv_dist', 'cd', 'n', 'x', 'y']; RULES = ('whole', 'centre_hull', 'centre_box', 'top_hull', 'top_box', 'top3_hull', 'top3_box')
NAMES = [f'{r}_{s}' for r in RULES for s in STATS]; PK = None


def stats(pts):
    if len(pts) < 16:
        return [np.nan] * 5
    try:
        vor = Voronoi(pts); tri = Delaunay(pts)
    except Exception:
        return [np.nan] * 5
    nb = {i: set() for i in range(len(pts))}
    for a, c in vor.ridge_points: nb[a].add(c); nb[c].add(a)
    A, S, D = [], [], []
    for i in range(len(pts)):
        g = vor.regions[vor.point_region[i]]
        if -1 in g or not g: continue
        v = vor.vertices[g]
        if (tri.find_simplex(v) < 0).any(): continue
        x, z = v[:, 0], v[:, 1]; A.append(0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))) * UM * UM); S.append(len(g)); D.append(np.mean([np.linalg.norm(pts[i] - pts[q]) for q in nb[i]]))
    if len(A) < 8:
        return [np.nan] * 4 + [float(len(A))]
    A, S, D = np.array(A), np.array(S), np.array(D)
    return [float((S == 6).mean()), float(A.std() / A.mean()), float(D.std() / D.mean()), float(1e6 / A.mean()), float(len(A))]


def windows(hi, w, h, k=3):
    """Top windows by peak count, stride 16, greedy non-overlap."""
    G = np.zeros((H // 16 + 2, W // 16 + 2)); ix = np.minimum((hi[:, 1] / 16).astype(int), G.shape[0] - 1); iy = np.minimum((hi[:, 0] / 16).astype(int), G.shape[1] - 1); np.add.at(G, (ix, iy), 1); I = np.zeros((G.shape[0] + 1, G.shape[1] + 1)); I[1:, 1:] = G.cumsum(0).cumsum(1)
    nh, nw = h // 16, w // 16; S = I[nh:, nw:] - I[:-nh, nw:] - I[nh:, :-nw] + I[:-nh, :-nw]; ymax = (H - h) // 16; xmax = (W - w) // 16; S = S[:ymax + 1, :xmax + 1]; out = []; S = S.copy()
    for _ in range(k):
        q = int(np.argmax(S)); y0, x0 = np.unravel_index(q, S.shape); out.append((int(x0) * 16, int(y0) * 16)); a0, a1 = max(0, y0 - nh // 2), min(S.shape[0], y0 + nh // 2 + 1); b0, b1 = max(0, x0 - nw // 2), min(S.shape[1], x0 + nw // 2 + 1); S[a0:a1, b0:b1] = -1
    return out


def one(r):
    pk = PK.get(r)
    if pk is None:
        return r, [np.nan] * len(NAMES)
    q = pk[pk[:, 2] >= 0.35, :2].astype(np.float64); hi = pk[pk[:, 2] >= 0.5, :2].astype(np.float64); out = stats(q) + [W / 2, H / 2]
    def inside(x0, y0, w, h): return q[(q[:, 0] >= x0) & (q[:, 0] < x0 + w) & (q[:, 1] >= y0) & (q[:, 1] < y0 + h)]
    for nm, (w, h) in SIZES.items():
        x0, y0 = (W - w) // 2, (H - h) // 2; out += stats(inside(x0, y0, w, h)) + [x0 + w / 2, y0 + h / 2]
    tops = {}
    for nm, (w, h) in SIZES.items():
        if len(hi) < 10:
            tops[nm] = None; out += [np.nan] * 7; continue
        ws = windows(hi, w, h); tops[nm] = ws; x0, y0 = ws[0]; out += stats(inside(x0, y0, w, h)) + [x0 + w / 2, y0 + h / 2]
    for nm, (w, h) in SIZES.items():
        if tops[nm] is None:
            out += [np.nan] * 7; continue
        S = np.array([stats(inside(x0, y0, w, h)) for x0, y0 in tops[nm]], float); W_ = np.nan_to_num(S[:, 4]); v = [float(np.nansum(S[:, j] * W_) / max(W_.sum(), 1e-9)) if W_.sum() > 0 else np.nan for j in range(4)] + [float(W_.sum())]; out += v + [np.nan, np.nan]
    return r, out


def main():
    global PK
    PK = T.load_peaks(); workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1))); F = np.full((8954, len(NAMES)), np.nan, np.float32)
    with get_context('fork').Pool(workers) as pool:
        for k, (r, v) in enumerate(pool.imap_unordered(one, sorted(PK), chunksize=16)):
            F[r] = v
            if (k + 1) % 2000 == 0: print(k + 1, flush=True)
    np.savez(OUT, F=F, names=np.array(NAMES)); print('done', workers, flush=True)


if __name__ == '__main__':
    main()
