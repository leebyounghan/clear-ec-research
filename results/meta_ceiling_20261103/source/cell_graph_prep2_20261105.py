"""Cell graphs with edge evidence (2026-11-05). Image tokens at nodes let the network memorise (+0.14 to +0.16), geometry alone cannot (-0.043 to -0.051). Edge features add low-dimensional physical evidence: is there a dark border where the Voronoi edge says two cells meet? For each Voronoi ridge, relative to the frame's median ridge response on such edges: mean and minimum border-map response along the ridge; dip between the two centres (border response at the midpoint relative to both ends, on the frame); ridge length / spacing; centre distance / spacing. Node additions: mean border response inside the cell relative to its edges (a real cell has dark edges and a clear inside). Everything label-free. Same windows and order as cell_graph_prep. Writes sweep/<OUT> with EA (edges x 5). usage: cell_graph_prep2_20261105.py OUT FAMILY_DIR K. Forked pool, clamp(16 - load1 - 1, 2, 12)."""
import os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
cv2.setNumThreads(0)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import abs_tess_20261104 as T
import legible_window_features_20261105 as L
W, H = 380, 300; ARG = sys.argv[1:] or ['cell_graphs_edge.npz', '', '3']; K = int(ARG[2]); OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/' + ARG[0]; PK = None
FH, FW = T.H, T.W


def sample(R, xy):
    x = np.clip(xy[..., 0], 0, FW - 1.001); y = np.clip(xy[..., 1], 0, FH - 1.001); x0 = x.astype(int); y0 = y.astype(int); dx = x - x0; dy = y - y0
    return (R[y0, x0] * (1 - dx) * (1 - dy) + R[y0, x0 + 1] * dx * (1 - dy) + R[y0 + 1, x0] * (1 - dx) * dy + R[y0 + 1, x0 + 1] * dx * dy)


def graph(q, x0, y0, R):
    p = q[:, :2].astype(np.float64); c = q[:, 2]; n = len(p)
    if n < 12:
        return None
    try:
        vor = Voronoi(p); tri = Delaunay(p)
    except Exception:
        return None
    a = np.full(n, np.nan); s = np.zeros(n); ub = np.zeros(n)
    for i in range(n):
        g = vor.regions[vor.point_region[i]]
        if -1 in g or not g: ub[i] = 1; s[i] = max(len(g) - 1, 3); continue
        v = vor.vertices[g]; s[i] = len(g)
        if (tri.find_simplex(v) < 0).any(): ub[i] = 1
        x, z = v[:, 0], v[:, 1]; a[i] = 0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1)))
    E = np.array(vor.ridge_points); RV = vor.ridge_vertices; d = np.linalg.norm(p[E[:, 0]] - p[E[:, 1]], axis=1); sp = float(np.median(d))
    cnt = np.bincount(E.ravel(), minlength=n).astype(float); dm = (np.bincount(E[:, 0], d, n) + np.bincount(E[:, 1], d, n)) / np.maximum(cnt, 1); cn = (np.bincount(E[:, 0], c[E[:, 1]], n) + np.bincount(E[:, 1], c[E[:, 0]], n)) / np.maximum(cnt, 1); nn = cKDTree(p).query(p, k=2)[0][:, 1]
    ts = np.linspace(0, 1, 9)[:, None]; mean_r = np.full(len(E), np.nan); min_r = np.full(len(E), np.nan); ln = np.zeros(len(E))
    for k, (rv, (i, j)) in enumerate(zip(RV, E)):
        if -1 in rv:
            mid = 0.5 * (p[i] + p[j]); seg = mid[None]; ln[k] = 0.0
        else:
            A, B_ = vor.vertices[rv[0]], vor.vertices[rv[1]]; ln[k] = np.linalg.norm(A - B_); seg = A * (1 - ts) + B_ * ts
        v = sample(R, seg); mean_r[k] = v.mean(); min_r[k] = np.percentile(v, 25)
    mids = sample(R, 0.5 * (p[E[:, 0]] + p[E[:, 1]])); ends = 0.5 * (sample(R, p[E[:, 0]]) + sample(R, p[E[:, 1]])); ref = max(float(np.nanmedian(mean_r)), 1e-3)
    EA = np.c_[np.nan_to_num(mean_r / ref).clip(0, 5), np.nan_to_num(min_r / ref).clip(0, 5), ((mids - ends) / ref).clip(-5, 5), (ln / sp).clip(0, 4), (d / sp).clip(0, 4)]
    rc = (np.bincount(E[:, 0], EA[:, 0], n) + np.bincount(E[:, 1], EA[:, 0], n)) / np.maximum(cnt, 1); inside = sample(R, p) / ref
    med_a = np.nanmedian(a[ub == 0]) if (ub == 0).sum() > 3 else np.nanmedian(a); F = np.c_[c, np.nan_to_num(np.log(np.maximum(a, 1) / max(med_a, 1)), nan=0.0, posinf=0.0, neginf=0.0).clip(-2, 2), np.log(dm / np.median(dm)).clip(-2, 2), np.log(nn / np.median(nn)).clip(-2, 2), s.clip(3, 9), cn, ub, (p[:, 0] - x0) / W, (p[:, 1] - y0) / H, rc.clip(0, 5), inside.clip(0, 5)]
    return F.astype(np.float32), E.astype(np.int32), EA.astype(np.float32)


def one(r):
    pk = PK.get(r)
    if pk is None:
        return r, []
    q = pk[pk[:, 2] >= 0.35]; hi = pk[pk[:, 2] >= 0.5, :2].astype(np.float64)
    if len(hi) < 10 or len(q) < 16:
        return r, []
    a = np.asarray(np.memmap(T.RAW, dtype=np.uint8, mode='r', shape=(8954, FH, FW))[r]); s = float(np.median(cKDTree(q[:, :2]).query(q[:, :2], k=2)[0][:, 1])); R = T.border_map(a, s); out = []
    for x0, y0 in L.windows(hi, W, H, k=K):
        m = (q[:, 0] >= x0) & (q[:, 0] < x0 + W) & (q[:, 1] >= y0) & (q[:, 1] < y0 + H); g = graph(q[m], x0, y0, R)
        if g is not None: out.append(g)
    return r, out


def main():
    global PK
    if ARG[1]:
        PK = {}
        for f in range(5):
            z = np.load(f'{ARG[1]}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
            for k_, i in enumerate(ix): PK[int(i)] = pk[of[k_]:of[k_ + 1]]
    else:
        PK = T.load_peaks()
    workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1))); res = [None] * 8954
    with get_context('fork').Pool(workers) as pool:
        for k, (r, g) in enumerate(pool.imap_unordered(one, range(8954), chunksize=16)):
            res[r] = g
            if (k + 1) % 2000 == 0: print(k + 1, flush=True)
    X, E_, EA, wid, offs, eoff = [], [], [], [], [0], [0]; n = 0; m = 0
    for r in range(8954):
        for w, (f, e, ea) in enumerate(res[r] or []):
            X.append(f); E_.append(e + n); EA.append(ea); wid.append(np.full(len(f), w, np.int8)); n += len(f); m += len(e)
        offs.append(n); eoff.append(m)
    np.savez(OUT, X=np.concatenate(X), E=np.concatenate(E_), EA=np.concatenate(EA), wid=np.concatenate(wid), offs=np.array(offs), eoff=np.array(eoff)); print('finished', workers, 'nodes', n, 'edges', m, flush=True)


if __name__ == '__main__':
    main()
