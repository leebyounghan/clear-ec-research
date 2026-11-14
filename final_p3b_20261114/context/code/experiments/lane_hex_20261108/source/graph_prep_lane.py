"""Cell graphs from the three most legible windows of every frame (2026-11-05). Nodes: pre-only peaks >= 0.35 in the window. Node features: confidence, log area / window median, log neighbour distance / median, log NN distance / median, side count (clipped 3-9), neighbour mean confidence, unbounded flag, position in window. Edges: Voronoi neighbours. Label-free. Writes sweep/<OUT>: concatenated nodes, edges, offsets per frame, window id per node. usage: cell_graph_prep3_20261105.py OUT FAMILY_DIR K [NODE_T] with K = 0 for the whole frame as one graph; NODE_T is the lowest peak confidence that becomes a node, default 0.35. Forked pool, clamp(16 - load1 - 1, 2, 12)."""
import os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, "/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source")
import abs_tess_20261104 as T
import legible_window_features_20261105 as L
W, H = int(os.environ.get("WW", "380")), int(os.environ.get("WH", "300")); ARG = sys.argv[1:] or ['cell_graphs.npz', '', '3']; K = int(ARG[2]); NODE_T = float(ARG[3]) if len(ARG) > 3 else 0.35; OUT = "/workspace/clear-ec-research/experiments/lane_hex_20261108/feats/" + ARG[0]; PK = None; FEATS = ['conf', 'log_area', 'log_dist', 'log_nn', 'sides', 'nb_conf', 'unbounded', 'x', 'y']


def graph(q, x0, y0):
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
    E = np.array(vor.ridge_points); d = np.linalg.norm(p[E[:, 0]] - p[E[:, 1]], axis=1); cnt = np.bincount(E.ravel(), minlength=n).astype(float); sm = np.bincount(E[:, 0], d, n) + np.bincount(E[:, 1], d, n); dm = sm / np.maximum(cnt, 1); cn = (np.bincount(E[:, 0], c[E[:, 1]], n) + np.bincount(E[:, 1], c[E[:, 0]], n)) / np.maximum(cnt, 1); nn = cKDTree(p).query(p, k=2)[0][:, 1]
    med_a = np.nanmedian(a[ub == 0]) if (ub == 0).sum() > 3 else np.nanmedian(a); F = np.c_[c, np.nan_to_num(np.log(np.maximum(a, 1) / max(med_a, 1)), nan=0.0, posinf=0.0, neginf=0.0).clip(-2, 2), np.log(dm / np.median(dm)).clip(-2, 2), np.log(nn / np.median(nn)).clip(-2, 2), s.clip(3, 9), cn, ub, (p[:, 0] - x0) / (W if K else 1296), (p[:, 1] - y0) / (H if K else 972)]
    return F.astype(np.float32), E.astype(np.int32), p.astype(np.float32)


def one(r):
    pk = PK.get(r)
    if pk is None:
        return r, []
    q = pk[pk[:, 2] >= NODE_T]; hi = pk[pk[:, 2] >= 0.5, :2].astype(np.float64)
    if len(hi) < 10 or len(q) < 16:
        return r, []
    if K == 0:
        g = graph(q, 0.0, 0.0); return r, [g] if g is not None else []
    out = []
    for x0, y0 in L.windows(hi, W, H, k=K):
        m = (q[:, 0] >= x0) & (q[:, 0] < x0 + W) & (q[:, 1] >= y0) & (q[:, 1] < y0 + H); g = graph(q[m], x0, y0)
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
    workers = int(os.environ.get("NW", "3")); res = [None] * 8954
    with get_context('fork').Pool(workers) as pool:
        for r, g in pool.imap_unordered(one, range(8954), chunksize=32): res[r] = g
    X, E_, wid, offs, eoff, XY = [], [], [], [0], [0], []; n = 0; m = 0
    for r in range(8954):
        for w, (f, e, xy) in enumerate(res[r] or []):
            XY.append(xy); X.append(f); E_.append(e + n); wid.append(np.full(len(f), w, np.int8)); n += len(f); m += len(e)
        offs.append(n); eoff.append(m)
    np.savez(OUT, xy=np.concatenate(XY), X=np.concatenate(X), E=np.concatenate(E_), wid=np.concatenate(wid), offs=np.array(offs), eoff=np.array(eoff), feats=np.array(FEATS)); print('done', workers, 'nodes', n, 'edges', m, 'frames with graphs', int((np.diff(offs) > 0).sum()), flush=True)


if __name__ == '__main__':
    main()
