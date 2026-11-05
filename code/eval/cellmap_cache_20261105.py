"""Label-free cell maps for every frame (2026-11-05), half resolution, uint8. Channel 0: dark-ridge response tied to the frame's cell spacing (abs_tess border map), scaled by its 99.5th percentile. Channel 1: pre-only peaks rendered as Gaussians (sigma 0.18 x spacing) with peak confidence as amplitude. Both computable at test time. Writes sweep/cellmaps_u8.dat, shape (8954, 2, 486, 648). Forked pool, clamp(16 - load1 - 1, 2, 12)."""
import os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import abs_tess_20261104 as T
OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/cellmaps_u8.dat'; H2, W2 = 486, 648; PK = None
cv2.setNumThreads(0)


def render(pk, s):
    g = np.zeros((H2, W2), np.float32); r = max(1, int(round(0.18 * s / 2 * 3))); sg = max(0.18 * s / 2, 0.7); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; K = np.exp(-(xx * xx + yy * yy) / (2 * sg * sg)).astype(np.float32)
    for x, y, p in pk:
        cx, cy = int(round(x / 2)), int(round(y / 2)); a0, a1, b0, b1 = max(0, cy - r), min(H2, cy + r + 1), max(0, cx - r), min(W2, cx + r + 1)
        if a0 >= a1 or b0 >= b1: continue
        v = K[a0 - cy + r:a1 - cy + r, b0 - cx + r:b1 - cx + r] * float(p); g[a0:a1, b0:b1] = np.maximum(g[a0:a1, b0:b1], v)
    return g


def one(i):
    a = np.asarray(np.memmap(T.RAW, dtype=np.uint8, mode='r', shape=(8954, T.H, T.W))[i]); pk = PK.get(i); pk = np.zeros((0, 3)) if pk is None else pk; q = pk[pk[:, 2] >= 0.35, :2]; s = float(np.median(cKDTree(q).query(q, k=2)[0][:, 1])) if len(q) >= 30 else 22.4
    R = T.border_map(a, s); R = cv2.resize(R, (W2, H2), interpolation=cv2.INTER_AREA); R = np.clip(R / max(float(np.percentile(R, 99.5)), 1e-3), 0, 1); G = render(pk, s)
    m = np.memmap(OUT, dtype=np.uint8, mode='r+', shape=(8954, 2, H2, W2)); m[i, 0] = (R * 255).astype(np.uint8); m[i, 1] = (np.clip(G, 0, 1) * 255).astype(np.uint8); m.flush(); return i


def main():
    global PK
    PK = T.load_peaks(); m = np.memmap(OUT, dtype=np.uint8, mode='w+', shape=(8954, 2, H2, W2)); del m; workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    with get_context('fork').Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(one, range(8954), chunksize=16)):
            if (k + 1) % 2000 == 0: print(k + 1, flush=True)
    print('done', workers, flush=True)


if __name__ == '__main__':
    main()
