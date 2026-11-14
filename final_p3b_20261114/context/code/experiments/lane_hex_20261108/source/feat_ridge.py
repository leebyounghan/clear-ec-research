"""Ridge evidence for HEX (HEX lane, 2026-11-08). For every window graph in a cell-graph file, recompute the Voronoi of its
nodes (same points, same order, so ridges match the graph's edges) and measure per ridge: length relative to the window's
median ridge, and border darkness = 1 - (mean image along the ridge) / (mean image at the two cell centres). Per node:
side counts that keep only ridges longer than tau x median (tau 0.15/0.25/0.35), side count of dark ridges, mean ridge
darkness. Per frame (row order): hex6 shares of interior cells under each count, averaged over windows by cell count.
usage: feat_ridge.py GRAPHS.npz OUTTAG   ->  feats/ridge_<OUTTAG>.npz (frame features) and feats/<OUTTAG>_ridge_graph.npz
(the graph file with 5 extra node columns: (s15-6)/2, (s25-6)/2, (s35-6)/2, (sdark-6)/2, darkness z)."""
import os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import Voronoi, Delaunay
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
GP, TAG = sys.argv[1], sys.argv[2]
Z = np.load(GP); XY, OFFS, WID, EOFF = Z['xy'], Z['offs'], Z['wid'], Z['eoff']
MM = None; TAUS = (0.15, 0.25, 0.35)


def samp(img, pts):
    x = np.clip(np.rint(pts[:, 0]).astype(int), 0, img.shape[1] - 1); y = np.clip(np.rint(pts[:, 1]).astype(int), 0, img.shape[0] - 1); return img[y, x]


def window(img, p):
    n = len(p); out = np.zeros((n, 5), np.float32); out[:, :4] = 6.0; interior = np.zeros(n, bool)
    try:
        vor = Voronoi(p.astype(np.float64)); tri = Delaunay(p.astype(np.float64))
    except Exception:
        return out, interior, 0
    RP = np.asarray(vor.ridge_points); RV = vor.ridge_vertices; fin = np.array([(-1 not in v) for v in RV]); nr = len(RP)
    V = vor.vertices; v1 = np.array([V[v[0]] if f else (0, 0) for v, f in zip(RV, fin)]); v2 = np.array([V[v[1]] if f else (0, 0) for v, f in zip(RV, fin)])
    L = np.where(fin, np.linalg.norm(v1 - v2, axis=1), np.nan); medL = np.nanmedian(L[fin]) if fin.any() else 1.0
    ts = np.linspace(0.2, 0.8, 5); seg = (v1[:, None, :] * (1 - ts[None, :, None]) + v2[:, None, :] * ts[None, :, None]).reshape(-1, 2)
    ir = samp(img, seg).reshape(nr, 5).mean(1); cc = np.zeros(n)
    for dx in (-2, 0, 2):
        for dy in (-2, 0, 2): cc += samp(img, p + np.array([dx, dy]))
    cc /= 9; dark = np.where(fin, 1 - ir / np.maximum(0.5 * (cc[RP[:, 0]] + cc[RP[:, 1]]), 1), np.nan)
    kd = np.nanmedian(dark[fin]) * 0.5 if fin.any() else 0.0
    for j, t in enumerate(TAUS):
        k = fin & (L >= t * medL); out[:, j] = np.bincount(RP[k].ravel(), minlength=n)
    k = fin & (dark >= kd); out[:, 3] = np.bincount(RP[k].ravel(), minlength=n)
    sm = np.bincount(RP[fin, 0], dark[fin], n) + np.bincount(RP[fin, 1], dark[fin], n); cnt = np.bincount(RP[fin].ravel(), minlength=n); out[:, 4] = sm / np.maximum(cnt, 1)
    for i in range(n):
        g = vor.regions[vor.point_region[i]]
        if g and -1 not in g and (tri.find_simplex(V[g]) >= 0).all(): interior[i] = True
    return out, interior, nr


def one(r):
    a, b = OFFS[r], OFFS[r + 1]
    if b - a == 0: return r, None, None, 0
    img = np.asarray(MM[r], dtype=np.float32); nodes = np.zeros((b - a, 5), np.float32); fr = []; nr_tot = 0
    for w in np.unique(WID[a:b]):
        m = np.where(WID[a:b] == w)[0]; o, it, nr = window(img, XY[a:b][m]); nodes[m] = o; nr_tot += nr
        if it.sum() >= 5: fr.append([it.sum()] + [np.mean(o[it, j] == 6) for j in range(4)] + [float(np.mean(o[it, 4]))])
    fr = np.array(fr) if fr else None
    return r, nodes, fr, nr_tot


def main():
    global MM
    MM = np.memmap('/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
    NODE = np.zeros((len(XY), 5), np.float32); FR = np.full((8954, 5), np.nan); bad = 0
    with get_context('fork').Pool(int(os.environ.get('NW', '2'))) as pool:
        for k, (r, nodes, fr, nr) in enumerate(pool.imap_unordered(one, range(8954), chunksize=16)):
            if nodes is not None: NODE[OFFS[r]:OFFS[r + 1]] = nodes
            if nr != EOFF[r + 1] - EOFF[r]: bad += 1
            if fr is not None: w = fr[:, 0]; FR[r] = (fr[:, 1:] * w[:, None]).sum(0) / w.sum()
            if k % 1000 == 0: print(k, flush=True)
    print('frames whose ridge count differs from the graph edges', bad, flush=True)
    names = np.array(['h6_s15', 'h6_s25', 'h6_s35', 'h6_dark', 'dark_mean'])
    np.savez_compressed(f'{LANE}/feats/ridge_{TAG}.npz', F=FR, names=names)
    ext = np.c_[(NODE[:, :4] - 6) / 2, (NODE[:, 4] - NODE[:, 4].mean()) / (NODE[:, 4].std() + 1e-9)].astype(np.float32)
    np.savez(f'{LANE}/feats/{TAG}_ridge_graph.npz', xy=XY, X=np.c_[Z['X'], ext], E=Z['E'], wid=WID, offs=OFFS, eoff=EOFF, feats=np.r_[Z['feats'], ['s15', 's25', 's35', 'sdark', 'darkz']])
    print('done', flush=True)


if __name__ == '__main__':
    main()
