"""Whole-frame features from the image-fitted tessellation for every frame (2026-11-04). Label-free, so the 8953 training labels are a
fair test: does topology read from image ridges carry signal beyond B and beyond the hex6 detector feature? Pilot on 25: HEX r 0.75-0.81
(expert oracle 0.84, Voronoi on the same peaks 0.39). Rows run in random order and a checkpoint is written every 500 frames.
usage: abs_features_20261104.py OUT_NPZ. Forked pool, clamp(16 - load1 - 1, 2, 12), OMP 1."""
import json, os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import abs_tess_20261104 as T
H, W = T.H, T.W
CONFIGS = (('t50', 0.5, 99.0), ('t35', 0.35, 99.0), ('t35s', 0.35, 0.6))
EDGES = (3, 5, 8)
STATS = ['n', 'cv', 'cv_trim', 'p95_50', 'p05_50', 'area_med', 'nb_mean', 'hex3', 'hex5', 'hex8', 'f5', 'f7', 'f4m', 'f8p']
REGIONS = ('all', 'centre', 'strong')
NAMES = [f'{c[0]}_{r}_{s}' for c in CONFIGS for r in REGIONS for s in STATS]
PK = None


def neighbours(lab, edges):
    ed = T.rag(lab); n = int(lab.max()); nb = {e: np.zeros(n + 1, int) for e in edges}; sup = np.zeros(n + 1); cnt = np.zeros(n + 1)
    return ed, nb, sup, cnt


def stats(ar, nbs):
    if len(ar) < 12:
        return [float(len(ar))] + [np.nan] * (len(STATS) - 1)
    med = np.median(ar); lo, hi = np.percentile(ar, (2, 98)); tr = ar[(ar >= lo) & (ar <= hi)]; n5 = nbs[5]
    return [float(len(ar)), float(ar.std() / ar.mean()), float(tr.std() / tr.mean()), float(np.percentile(ar, 95) / med), float(np.percentile(ar, 5) / med), float(med), float(n5.mean()),
            float((nbs[3] == 6).mean()), float((n5 == 6).mean()), float((nbs[8] == 6).mean()), float((n5 == 5).mean()), float((n5 == 7).mean()), float((n5 <= 4).mean()), float((n5 >= 8).mean())]


def one(r):
    a = np.asarray(np.memmap(T.RAW, dtype=np.uint8, mode='r', shape=(8954, H, W))[r]); pk = PK[r]; out = []
    q = pk[pk[:, 2] >= 0.35, :2]
    if len(q) < 30:
        return r, [np.nan] * len(NAMES)
    s = float(np.median(cKDTree(q).query(q, k=2)[0][:, 1]))
    for tag, t0, ts in CONFIGS:
        lab, info = T.tessellate(a, pk, t0, 0.0, ts, spacing=s)
        n = int(lab.max()); idx = np.arange(1, n + 1); area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(float)
        cy = np.bincount(np.nonzero(lab)[0], minlength=0) if False else None
        yy, xx = np.nonzero(lab); ll = lab[yy, xx]; cy = np.bincount(ll, yy, n + 1)[1:] / np.maximum(area, 1); cx = np.bincount(ll, xx, n + 1)[1:] / np.maximum(area, 1)
        ed = T.rag(lab); nb = {e: np.zeros(n + 1, int) for e in EDGES}; R = T.border_map(a, s); weak = np.zeros(n + 1, bool); st = [float(np.mean(R[v])) for v in ed.values()]; ref = np.median(st) if st else 1.0
        for (i, j), v in ed.items():
            for e in EDGES:
                if len(v[0]) >= e: nb[e][i] += 1; nb[e][j] += 1
            if len(v[0]) >= 5 and np.mean(R[v]) < 0.6 * ref: weak[i] = True; weak[j] = True
        touch = np.zeros(n + 1, bool); bgd = T.ndi.binary_dilation(lab == 0); touch[np.unique(lab[bgd])] = True; touch[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = True; touch[0] = False
        ok = ~touch[1:] & (area > 0); nbs = {e: nb[e][1:] for e in EDGES}
        cen = (cx > 0.2 * W) & (cx < 0.8 * W) & (cy > 0.2 * H) & (cy < 0.8 * H)
        for m in (ok, ok & cen, ok & ~weak[1:]):
            out += stats(area[m], {e: nbs[e][m] for e in EDGES})
    return r, out


def main():
    global PK
    out = sys.argv[1]; PK = T.load_peaks(); rows = np.random.default_rng(0).permutation(sorted(PK)); F = np.full((8954, len(NAMES)), np.nan, np.float32); seen = np.zeros(8954, bool)
    workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    with get_context('fork').Pool(workers) as pool:
        for k, (r, v) in enumerate(pool.imap_unordered(one, rows.tolist(), chunksize=4)):
            F[r] = v; seen[r] = True
            if (k + 1) % 500 == 0:
                np.savez(out + '.tmp.npz', F=F, seen=seen, names=np.array(NAMES)); os.replace(out + '.tmp.npz', out); print(k + 1, flush=True)
    np.savez(out + '.tmp.npz', F=F, seen=seen, names=np.array(NAMES)); os.replace(out + '.tmp.npz', out); print('done', workers, flush=True)


if __name__ == '__main__':
    main()
