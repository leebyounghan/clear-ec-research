"""CV lane (2026-11-08). Cell graphs as cell_graph_prep3_20261105 (same node features, edges, windows from peaks >= 0.5), but the nodes are
re-selected with a per-image spacing prior: s from predicted CD (hybrid B, OOF), greedy NMS at alpha*s on peaks >= t_hi, optional fill with
peaks in [t_lo, t_hi) farther than beta*s from kept ones. Also writes per-frame window statistics (area CV, log-area SD, NN CV, edge CV) of the nodes.
usage: graph_prep_sp_20261108.py OUT FAMILY_DIR K W H T_HI ALPHA [T_LO BETA]   (FAMILY_DIR '' = shipped pre-only). 4 workers."""
import os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import abs_tess_20261104 as T
import legible_window_features_20261105 as L
import ceiling_meta_cpu_20261103 as C
A = sys.argv[1:]; OUTN, FAM, K, W, H, T_HI, ALPHA = A[0], A[1], int(A[2]), int(A[3]), int(A[4]), float(A[5]), float(A[6]); T_LO = float(A[7]) if len(A) > 7 else None; BETA = float(A[8]) if len(A) > 8 else 0.8
OUT = '/workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/' + OUTN; PK = None; SP = None; FEATS = ['conf', 'log_area', 'log_dist', 'log_nn', 'sides', 'nb_conf', 'unbounded', 'x', 'y']; UMPX = 0.7716


def select(pk, s):
    q = pk[pk[:, 2] >= T_HI]; q = q[np.argsort(-q[:, 2])]
    if ALPHA > 0 and len(q):
        keep = np.ones(len(q), bool); tree = cKDTree(q[:, :2]); r = ALPHA * s
        for i in range(len(q)):
            if not keep[i]: continue
            for j in tree.query_ball_point(q[i, :2], r):
                if j > i: keep[j] = False
        q = q[keep]
    if T_LO is not None and T_LO < T_HI:
        c = pk[(pk[:, 2] >= T_LO) & (pk[:, 2] < T_HI)]; c = c[np.argsort(-c[:, 2])]; add = []
        pts = [tuple(v) for v in q[:, :2]]; tree = cKDTree(q[:, :2]) if len(q) else None; extra = []
        for i in range(len(c)):
            d1 = tree.query(c[i, :2])[0] if tree is not None else 1e9
            d2 = np.min(np.linalg.norm(np.asarray(extra) - c[i, :2], axis=1)) if extra else 1e9
            if min(d1, d2) >= BETA * s: extra.append(c[i, :2]); add.append(c[i])
        if add: q = np.r_[q, np.asarray(add)]
    return q


def graph(q, x0, y0):
    p = q[:, :2].astype(np.float64); c = q[:, 2]; n = len(p)
    if n < 12: return None
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
    ar = a[(ub == 0) & np.isfinite(a)]; dd = d[d < 3 * np.median(d)]
    st = [ar.std() / ar.mean(), np.log(ar).std(), nn.std() / nn.mean(), dd.std() / dd.mean()] if len(ar) >= 10 else [np.nan] * 4
    return F.astype(np.float32), E.astype(np.int32), p.astype(np.float32), st


def one(r):
    pk = PK.get(r)
    if pk is None: return r, [], [np.nan] * 4
    hi = pk[pk[:, 2] >= 0.5, :2].astype(np.float64)
    if len(hi) < 10: return r, [], [np.nan] * 4
    q = select(pk, SP[r])
    if len(q) < 16: return r, [], [np.nan] * 4
    if K == 0:
        g = graph(q, 0.0, 0.0); return (r, [g[:3]], g[3]) if g is not None else (r, [], [np.nan] * 4)
    out = []; sts = []
    for x0, y0 in L.windows(hi, W, H, k=K):
        m = (q[:, 0] >= x0) & (q[:, 0] < x0 + W) & (q[:, 1] >= y0) & (q[:, 1] < y0 + H); g = graph(q[m], x0, y0)
        if g is not None: out.append(g[:3]); sts.append(g[3])
    return r, out, (np.nanmean(np.array(sts), 0).tolist() if sts else [np.nan] * 4)


def main():
    global PK, SP
    b = C.E.load_base(); idx = b['idx'].astype(int); B = b['B']; sp = np.sqrt(2e6 / (np.sqrt(3) * B[:, 0])) / UMPX; SP = np.full(8954, np.nan); SP[idx] = sp
    if FAM:
        PK = {}
        for f in range(5):
            z = np.load(f'{FAM}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
            for k_, i in enumerate(ix): PK[int(i)] = pk[of[k_]:of[k_ + 1]]
    else:
        PK = T.load_peaks()
    res = [None] * 8954; ST = np.full((8954, 4), np.nan)
    with get_context("fork").Pool(int(os.environ.get("PREP_WORKERS", "4"))) as pool:
        for r, g, st in pool.imap_unordered(one, range(8954), chunksize=32): res[r] = g; ST[r] = st
    X, E_, wid, offs, eoff, XY = [], [], [], [0], [0], []; n = 0; m = 0
    for r in range(8954):
        for w, (f, e, xy) in enumerate(res[r] or []):
            XY.append(xy); X.append(f); E_.append(e + n); wid.append(np.full(len(f), w, np.int8)); n += len(f); m += len(e)
        offs.append(n); eoff.append(m)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, xy=np.concatenate(XY), X=np.concatenate(X), E=np.concatenate(E_), wid=np.concatenate(wid), offs=np.array(offs), eoff=np.array(eoff), feats=np.array(FEATS), win_stats=ST, win_stat_names=np.array(['area_cv', 'log_area_sd', 'nn_cv', 'edge_cv']))
    print('done nodes', n, 'edges', m, 'frames with graphs', int((np.diff(offs) > 0).sum()), 'nodes/frame', round(n / max(1, int((np.diff(offs) > 0).sum())), 1), flush=True)


if __name__ == '__main__':
    main()
