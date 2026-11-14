"""Per-node appearance for the cell graph (HEX lane, 2026-11-08): centre brightness / window mean, border darkness on a ring at
0.45 x nearest-neighbour distance (8 directions), ring spread, ring minimum. Appended (z-scored, clipped +-5) to a graph file.
usage: feat_nodeapp.py IN_GRAPH.npz OUT_NAME.npz   (both in feats/)."""
import os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import cKDTree
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
Z = np.load(f'{LANE}/feats/{sys.argv[1]}'); OUT = sys.argv[2]; XY, OFFS, WID = Z['xy'], Z['offs'], Z['wid']; MM = None
ANG = np.linspace(0, 2 * np.pi, 8, endpoint=False); DIR = np.c_[np.cos(ANG), np.sin(ANG)]


def samp(img, pts):
    x = np.clip(np.rint(pts[..., 0]).astype(int), 0, img.shape[1] - 1); y = np.clip(np.rint(pts[..., 1]).astype(int), 0, img.shape[0] - 1); return img[y, x]


def one(r):
    a, b = OFFS[r], OFFS[r + 1]
    if b - a == 0: return r, None
    img = np.asarray(MM[r], dtype=np.float32); out = np.zeros((b - a, 4), np.float32)
    for w in np.unique(WID[a:b]):
        m = np.where(WID[a:b] == w)[0]; p = XY[a:b][m].astype(np.float64)
        if len(p) < 3: continue
        d, _ = cKDTree(p).query(p, k=2); nn = d[:, 1]
        cc = np.zeros(len(p))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1): cc += samp(img, p + np.array([dx, dy]))
        cc /= 9; ring = samp(img, p[:, None, :] + 0.45 * nn[:, None, None] * DIR[None]); rm = ring.mean(1)
        lo, hi = p.min(0).astype(int), p.max(0).astype(int) + 1; wm = img[lo[1]:hi[1], lo[0]:hi[0]].mean() + 1
        out[m] = np.c_[cc / wm, 1 - rm / np.maximum(cc, 1), ring.std(1) / np.maximum(rm, 1), ring.min(1) / np.maximum(cc, 1)]
    return r, out


def main():
    global MM
    MM = np.memmap('/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
    A = np.zeros((len(XY), 4), np.float32)
    with get_context('fork').Pool(int(os.environ.get('NW', '2'))) as pool:
        for k, (r, o) in enumerate(pool.imap_unordered(one, range(8954), chunksize=16)):
            if o is not None: A[OFFS[r]:OFFS[r + 1]] = o
            if k % 2000 == 0: print(k, flush=True)
    A = np.clip((A - A.mean(0)) / (A.std(0) + 1e-9), -5, 5).astype(np.float32)
    np.savez(f'{LANE}/feats/{OUT}', xy=XY, X=np.c_[Z['X'], A], E=Z['E'], wid=WID, offs=OFFS, eoff=Z['eoff'], feats=np.r_[Z['feats'], ['cbright', 'ringdark', 'ringspread', 'ringmin']])
    print('done', A.shape, flush=True)


if __name__ == '__main__':
    main()
