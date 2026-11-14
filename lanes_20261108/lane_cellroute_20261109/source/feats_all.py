"""Cell-route lane (2026-11-09): Konan k=2 / Voronoi k=0 cell statistics on all 8954 frames from cleaned detector point sets.
Point sets (chosen on the 25 marker frames, gate25 logs): P0 seeds3 TTA peaks >= 0.2 raw; P1 same + clean (merge < 0.5 s, fill Delaunay holes
cR 0.85, m <= 3); P2 vote of s46/s47/s48 TTA peaks >= 0.35 and seg-mask centroids (>= 1 source, i.e. union with cross-source merge); P3 = P2 + clean.
Cells: Delaunay over the whole frame, interior seeds; regions: 'win' = seeds inside the 3 legible 380x300 windows (peaks >= 0.5, as the CV lane),
'full' = all interior seeds, 'creg' (P2/P3 only) = seeds whose own and neighbours' vote share >= 0.5.
Columns per (set, region): cv_k0, cv_k2, hex_k0, hex_k2, n. Detector peaks are the pre-only fold detectors (OOF-safe); masks are zero-shot.
usage: feats_all.py [WORKERS]; writes feats/cellroute_feats.npz (dataset index order)."""
import os, sys, time
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.spatial import cKDTree
sys.path[:0] = ['/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source', '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source']
import cr_lib as L
import legible_window_features_20261105 as LW
ROOT = '/workspace/clear-ec-research/experiments/'; N = 8954


def load(tag):
    out = {}
    for f in range(5):
        z = np.load(ROOT + f'lane_cv_20261108/out/tta_peaks/{tag}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']  # members read once
        for k, i in enumerate(ix): out[int(i)] = pk[of[k]:of[k + 1]]
    return out


PK = {t: load(t) for t in ('seeds3_tta4', 's46_tta4', 's47_tta4', 's48_tta4')}
z = np.load(ROOT + 'lane_cellroute_20261109/feats/seg_centroids.npz'); _pts, _of, _ix = z['pts'], z['offsets'], z['indices']; SEG = {int(d): _pts[_of[k]:_of[k + 1]] for k, d in enumerate(_ix)}
SETS = ('P0', 'P1', 'P2', 'P3'); REG = {'P0': ('win', 'full'), 'P1': ('win', 'full'), 'P2': ('win', 'full', 'creg'), 'P3': ('win', 'full', 'creg')}
COLS = [f'{s}_{r}_{c}' for s in SETS for r in REG[s] for c in ('cv_k0', 'cv_k2', 'hex_k0', 'hex_k2', 'n')]


def stats(ix, res, m):
    sel = m[ix] if len(ix) else np.zeros(0, bool)
    if sel.sum() < 10: return [np.nan] * 4 + [float(sel.sum())]
    a0, s0 = res[0][0][sel], res[0][1][sel]; a2, s2 = res[2][0][sel], res[2][1][sel]
    return [a0.std() / a0.mean(), a2.std() / a2.mean(), float(np.mean(s0 == 6)), float(np.mean(s2 == 6)), float(sel.sum())]


def one(r):
    out = [np.nan] * len(COLS)
    pk = PK['seeds3_tta4'].get(r)
    if pk is None or len(pk) < 20: return r, out
    pk = pk.astype(float); hi = pk[pk[:, 2] >= 0.5, :2]; mid = pk[pk[:, 2] >= 0.35, :2]
    ref = hi if len(hi) > 20 else mid
    if len(ref) < 10: return r, out
    s = float(np.median(cKDTree(ref).query(ref, k=2)[0][:, 1])); wins = LW.windows(ref, 380, 300, k=3)
    q = pk[pk[:, 2] >= 0.2]; P0 = (q[:, :2], q[:, 2])
    P1 = L.clean(q[:, :2].copy(), q[:, 2].copy())[:2]
    srcs = [PK[t][r].astype(float) for t in ('s46_tta4', 's47_tta4', 's48_tta4')]; srcs = [x[x[:, 2] >= 0.35] for x in srcs]
    sg = SEG.get(r, np.zeros((0, 3))).astype(float); sg = sg[sg[:, 2] >= 0.15 * s ** 2]; srcs.append(np.c_[sg[:, :2], np.full(len(sg), 0.5)])
    P2 = L.vote(srcs, s, 0.45, 1); P3 = L.clean(P2[0].copy(), P2[1].copy())[:2]
    vals = []
    for nm, (p, w) in zip(SETS, (P0, P1, P2, P3)):
        ix, res, nb = L.konan_cells_idx(p)
        inw = np.zeros(len(p), bool)
        for x0, y0 in wins: inw |= (p[:, 0] >= x0) & (p[:, 0] < x0 + 380) & (p[:, 1] >= y0) & (p[:, 1] < y0 + 300)
        masks = {'win': inw, 'full': np.ones(len(p), bool)}
        if 'creg' in REG[nm]:
            cm = np.zeros(len(p), bool)
            if nb is not None: cm = np.array([w[i] >= 0.5 and (w[nb[i]] >= 0.5).all() for i in range(len(p))])
            masks['creg'] = cm
        for rg in REG[nm]: vals += stats(ix, res, masks[rg])
    return r, vals


if __name__ == '__main__':
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 3; F = np.full((N, len(COLS)), np.nan); t0 = time.time(); done = 0
    with get_context('fork').Pool(W) as pool:
        for r, v in pool.imap_unordered(one, range(N), chunksize=8):
            F[r] = v; done += 1
            if done % 500 == 0: print(done, 'frames', round((time.time() - t0) / 60, 1), 'min', flush=True)
    np.savez_compressed(ROOT + 'lane_cellroute_20261109/feats/cellroute_feats.npz', F=F, names=np.array(COLS), note=__doc__)
    print('done', int(np.isfinite(F[:, COLS.index('P1_full_cv_k2')]).sum()), 'frames with P1 full', flush=True)
