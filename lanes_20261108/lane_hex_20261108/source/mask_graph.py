"""Cell graphs from Cellpose-SAM masks (HEX lane, 2026-11-08), same file layout as graph_prep_lane.py so gnn_hex.py reads it.
Windows: the three most legible 380x300 windows from the pre46 peaks (as cell_graphs.npz). Nodes: mask cells (area > 30) with
centroid in the window. Edges: mask adjacency within 3 px. Node features (9): interior flag, log area / window median,
log mean neighbour distance / median, log NN distance / median, mask side count (clip 3-9), share of interior neighbours,
not-interior flag, x, y in the window. usage: mask_graph.py CHUNKDIR(rel LANE) OUT(name in feats/) [workers]"""
import os, sys, glob
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import abs_tess_20261104 as T
import legible_window_features_20261105 as LW
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
W, H = 380, 300; R = 3; OFFS = [(dy, dx) for dy in range(-R, R + 1) for dx in range(0, R + 1) if (dx > 0 or dy > 0) and dy * dy + dx * dx <= R * R]
FEATS = ['interior', 'log_area', 'log_dist', 'log_nn', 'sides', 'nb_int', 'not_int', 'x', 'y']
PK = None


def cells(M):
    M = M.astype(np.int32); Hh, Ww = M.shape; n = int(M.max()); pairs = []
    for dy, dx in OFFS:
        y0, y1 = max(0, -dy), Hh - max(0, dy); a = M[y0:y1, 0:Ww - dx]; b = M[y0 + dy:y1 + dy, dx:Ww]
        k = (a > 0) & (b > 0) & (a != b); pairs.append(np.c_[np.minimum(a[k], b[k]), np.maximum(a[k], b[k])])
    P = np.unique(np.concatenate(pairs), axis=0)
    area = np.bincount(M.ravel(), minlength=n + 1).astype(float); edge = np.unique(np.r_[M[0], M[-1], M[:, 0], M[:, -1]])
    dil = ndi.grey_dilation(M, size=(2 * R + 1, 2 * R + 1)); ring = (M == 0) & (dil > 0)
    bg = np.bincount(dil[ring], minlength=n + 1).astype(float); per = np.bincount(dil[(dil > 0) & (dil != M)], minlength=n + 1).astype(float)
    ok = np.ones(n + 1, bool); ok[0] = False; ok[edge] = False; ok &= bg / np.maximum(per, 1) < 0.25; ok &= area > 30
    lab = np.arange(n + 1); cy = np.nan_to_num(ndi.mean(np.indices(M.shape)[0], M, lab)); cx = np.nan_to_num(ndi.mean(np.indices(M.shape)[1], M, lab))
    return P, area, ok, cx, cy


def do_chunk(path):
    z = np.load(path); return [one(int(r), M) for r, M in zip(z['frames'], z['masks'])]


def one(r, M):
    pk = PK.get(r)
    if pk is None: return r, []
    hi = pk[pk[:, 2] >= 0.5, :2].astype(np.float64)
    if len(hi) < 10: return r, []
    if M.max() < 12: return r, []
    P, area, ok, cx, cy = cells(M); out = []
    for x0, y0 in LW.windows(hi, W, H, k=3):
        sel = np.where((area > 30) & (cx >= x0) & (cx < x0 + W) & (cy >= y0) & (cy < y0 + H))[0]; sel = sel[sel > 0]
        if len(sel) < 12: continue
        loc = -np.ones(len(area), np.int64); loc[sel] = np.arange(len(sel))
        e = P[(loc[P[:, 0]] >= 0) & (loc[P[:, 1]] >= 0)]; E = np.c_[loc[e[:, 0]], loc[e[:, 1]]]
        if len(E) < 12: continue
        n = len(sel); p = np.c_[cx[sel], cy[sel]]; a = area[sel]; o = ok[sel].astype(float)
        d = np.linalg.norm(p[E[:, 0]] - p[E[:, 1]], axis=1); cnt = np.bincount(E.ravel(), minlength=n).astype(float)
        dm = (np.bincount(E[:, 0], d, n) + np.bincount(E[:, 1], d, n)) / np.maximum(cnt, 1); dm[cnt == 0] = np.median(d)
        ni = (np.bincount(E[:, 0], o[E[:, 1]], n) + np.bincount(E[:, 1], o[E[:, 0]], n)) / np.maximum(cnt, 1)
        nn = cKDTree(p).query(p, k=2)[0][:, 1]; med_a = np.median(a[o > 0]) if (o > 0).sum() > 3 else np.median(a)
        F = np.c_[o, np.log(a / med_a).clip(-2, 2), np.log(dm / np.median(dm)).clip(-2, 2), np.log(nn / np.median(nn)).clip(-2, 2), cnt.clip(3, 9), ni, 1 - o, (p[:, 0] - x0) / W, (p[:, 1] - y0) / H]
        out.append((F.astype(np.float32), E.astype(np.int32), p.astype(np.float32)))
    return r, out


def main():
    global PK
    PK = T.load_peaks(); paths = sorted(glob.glob(f'{LANE}/{sys.argv[1]}/c[0-9][0-9][0-9].npz'))[:int(sys.argv[4]) if len(sys.argv) > 4 else None]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3; res = [None] * 8954
    with get_context('fork').Pool(workers) as pool:
        for out in pool.imap_unordered(do_chunk, paths, chunksize=1):
            for r, g in out: res[r] = g
    X, E_, wid, offs, eoff, XY = [], [], [], [0], [0], []; n = 0; m = 0
    for r in range(8954):
        for w, (f, e, xy) in enumerate(res[r] or []):
            XY.append(xy); X.append(f); E_.append(e + n); wid.append(np.full(len(f), w, np.int8)); n += len(f); m += len(e)
        offs.append(n); eoff.append(m)
    np.savez(f'{LANE}/feats/{sys.argv[2]}', xy=np.concatenate(XY), X=np.concatenate(X), E=np.concatenate(E_), wid=np.concatenate(wid), offs=np.array(offs), eoff=np.array(eoff), feats=np.array(FEATS))
    print('done', workers, 'nodes', n, 'edges', m, 'frames with graphs', int((np.diff(offs) > 0).sum()), flush=True)


if __name__ == '__main__':
    main()
