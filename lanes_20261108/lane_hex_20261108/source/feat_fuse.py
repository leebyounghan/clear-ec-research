"""Consensus points from many detectors -> hex6 variants (HEX lane, 2026-11-08). Per frame: pool peaks >= T of D detectors,
link peaks closer than R * spacing (connected components), a cluster = one cell with votes = distinct detectors, position =
confidence-weighted mean. For vote thresholds v, hex6 (bounded Voronoi cells inside the hull) over the whole frame and over the
three legible windows (pre-only s46 peaks >= 0.5, 380x300), plus a spacing-filtered hex6 that keeps only cells whose own and
neighbours' areas are within [0.55, 1.6] x median. Also per-detector hex6 for reference. usage: feat_fuse.py OUT SET [T] [R]"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import get_context
import numpy as np
from scipy.spatial import Voronoi, Delaunay, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import legible_window_features_20261105 as LW
G = '/workspace/clear-ec-research/experiments/green_track_20261028/'
SETS = {
 'pre3': ['external_srep/preonly', 'external_srep/s47_preonly', 'external_srep/s48_preonly'],
 'sg9': ['sg2_specialist'] + [f'sg2_specialist_s{s}' for s in range(47, 53)] + ['sg2_sigma3p5', 'sg2_sigma5'],
 'frz4': ['frozen_heads/sg2f_s46', 'frozen_heads/sg2f_s47', 'frozen_heads/sg2f_s48', 'frozen_heads/sg2pref_s46'],
}
SETS['all16'] = SETS['pre3'] + SETS['sg9'] + SETS['frz4']
SETS['prefrz'] = SETS['pre3'] + SETS['frz4']
OUT = sys.argv[1]; SET = sys.argv[2]; T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35; R = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
DETS = SETS[SET]; D = len(DETS); VS = sorted(set([1, 2, max(2, D // 4), max(2, D // 3), max(2, D // 2), max(2, (2 * D) // 3)]))
PK = []; PRE = {}


def load(d):
    out = {}
    for f in range(5):
        z = np.load(f'{G}{d}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
        for k, i in enumerate(ix):
            q = pk[of[k]:of[k + 1]]; out[int(i)] = q[q[:, 2] >= min(T, 0.35)]
    return out


def cells(p):
    """sides and areas of bounded Voronoi cells inside the hull; returns (sides, area, bounded mask, ridge pairs)."""
    n = len(p); s = np.zeros(n, int); a = np.full(n, np.nan); ok = np.zeros(n, bool)
    vor = Voronoi(p); tri = Delaunay(p)
    for i in range(n):
        g = vor.regions[vor.point_region[i]]
        if -1 in g or not g: continue
        v = vor.vertices[g]
        if (tri.find_simplex(v) < 0).any(): continue
        ok[i] = True; s[i] = len(g); x, z = v[:, 0], v[:, 1]; a[i] = 0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1)))
    return s, a, ok, np.array(vor.ridge_points)


def hexstats(p):
    """[hex6, frac5, frac7, n_bounded, hex6 spacing-filtered, n filtered]"""
    if len(p) < 20: return [np.nan] * 6
    try: s, a, ok, E = cells(p)
    except Exception: return [np.nan] * 6
    if ok.sum() < 12: return [np.nan] * 6
    h6 = (s[ok] == 6).mean(); f5 = (s[ok] == 5).mean(); f7 = (s[ok] == 7).mean()
    med = np.nanmedian(a[ok]); r = a / med; bad = ~ok | (r < 0.55) | (r > 1.6)
    nb_bad = np.zeros(len(p), bool); m = bad[E[:, 0]] | bad[E[:, 1]]
    nb_bad[E[m, 0]] = True; nb_bad[E[m, 1]] = True
    keep = ok & ~bad & ~(nb_bad & False)
    keep2 = ok & ~bad
    # cell is clean if itself is plausible and no neighbour is implausible
    adj_bad = np.zeros(len(p), bool); adj_bad[E[bad[E[:, 1]], 0]] = True; adj_bad[E[bad[E[:, 0]], 1]] = True
    clean = keep2 & ~adj_bad
    hc = (s[clean] == 6).mean() if clean.sum() >= 8 else np.nan
    return [h6, f5, f7, float(ok.sum()), hc, float(clean.sum())]


def fuse(qs, spacing):
    P = np.concatenate([q[:, :2] for q in qs]); C = np.concatenate([q[:, 2] for q in qs]); DI = np.concatenate([np.full(len(q), k) for k, q in enumerate(qs)])
    if len(P) < 20: return None
    pr = cKDTree(P).query_pairs(R * spacing, output_type='ndarray')
    n = len(P); A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)) if len(pr) else coo_matrix((n, n))
    nc, lab = connected_components(A, directed=False)
    w = C.astype(np.float64); sw = np.bincount(lab, w, nc); x = np.bincount(lab, w * P[:, 0], nc) / sw; yy = np.bincount(lab, w * P[:, 1], nc) / sw
    votes = np.zeros(nc, int); u = np.unique(lab * 64 + DI); np.add.at(votes, u // 64, 1)
    return np.c_[x, yy], votes, np.bincount(lab, C, nc) / np.bincount(lab, minlength=nc)


def one(r):
    qs = [pk.get(r) for pk in PK]
    if any(q is None for q in qs): return r, None
    qs = [q[q[:, 2] >= T] for q in qs]
    pre = PRE.get(r); hi = pre[pre[:, 2] >= 0.5, :2].astype(np.float64) if pre is not None else None
    out = {}
    for k, q in enumerate(qs): out[f'det{k}'] = hexstats(q[:, :2].astype(np.float64))[0]
    big = max(qs, key=len)
    if len(big) < 20: return r, out
    spacing = float(np.median(cKDTree(big[:, :2]).query(big[:, :2], k=2)[0][:, 1]))
    fz = fuse(qs, spacing)
    if fz is None: return r, out
    P, votes, mc = fz; out['spacing'] = spacing
    wins = LW.windows(hi, 380, 300, k=3) if hi is not None and len(hi) >= 10 else []
    for v in VS:
        p = P[votes >= v]; out[f'v{v}_n'] = float(len(p))
        hs = hexstats(p)
        for nm, val in zip(('h6', 'f5', 'f7', 'nb', 'h6c', 'nc'), hs): out[f'v{v}_{nm}'] = val
        ws = []
        for x0, y0 in wins:
            m = (p[:, 0] >= x0) & (p[:, 0] < x0 + 380) & (p[:, 1] >= y0) & (p[:, 1] < y0 + 300); ws.append(hexstats(p[m]))
        ws = np.array(ws) if ws else np.full((1, 6), np.nan)
        with np.errstate(all='ignore'):
            out[f'v{v}_wh6'] = float(np.nanmean(ws[:, 0])) if np.isfinite(ws[:, 0]).any() else np.nan
            out[f'v{v}_wh6c'] = float(np.nanmean(ws[:, 4])) if np.isfinite(ws[:, 4]).any() else np.nan
            out[f'v{v}_w1h6'] = float(ws[0, 0])
    return r, out


def main():
    global PK, PRE
    PK = [load(d) for d in DETS]; PRE = PK[0] if DETS[0] == 'external_srep/preonly' else load('external_srep/preonly')
    frames = sorted(set.intersection(*[set(p) for p in PK]))
    print('dets', D, 'frames', len(frames), 'votes', VS, flush=True)
    res = {}
    with get_context('fork').Pool(int(os.environ.get('NW', '3'))) as pool:
        for k, (r, o) in enumerate(pool.imap_unordered(one, frames, chunksize=16)):
            res[r] = o
            if k % 2000 == 0: print(k, flush=True)
    keys = sorted({k for o in res.values() if o for k in o})
    F = np.full((8954, len(keys)), np.nan, np.float32)
    for r, o in res.items():
        if o:
            for j, k in enumerate(keys): F[r, j] = o.get(k, np.nan)
    np.savez_compressed(OUT, F=F, names=np.array(keys), dets=np.array(DETS), T=T, R=R); print('done', F.shape, flush=True)


if __name__ == '__main__':
    main()
