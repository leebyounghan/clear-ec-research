"""Cell-route lane (2026-11-09): Konan k=0/k=2 statistics on all 8954 frames with operator-like cell selection (diag_box.py / diag_region.py gate).
Point sets: P2 = vote of s46/s47/s48 TTA peaks >= 0.35 + seg-mask centroids (>= 1 source); P0 = seeds3 TTA peaks >= 0.2. Cells: Delaunay over the
whole frame, interior seeds. Regions: box = the 538x408 window (technician box size) with the most peaks >= 0.5; p150 = the 150 interior cells in the box
nearest the mean of its confident peaks; win = 3 legible 380x300 windows; full. Each region raw and trimmed (cells with area outside [0.4, 2.5] x the region's
median dropped; bound picked once, not tuned). Columns per set/region/trim: cv_k0, cv_k2, hex_k0, hex_k2, n.
usage: feats_trim.py WORKERS; writes feats/trim_feats.npz (dataset index order)."""
import os, sys, time
from multiprocessing import get_context
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from scipy.spatial import cKDTree
sys.path[:0] = ['/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source', '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source']
import cr_lib as L
import legible_window_features_20261105 as LW
ROOT = '/workspace/clear-ec-research/experiments/'; N = 8954; BW, BH = 538, 408


def load(tag):
    out = {}
    for f in range(5):
        z = np.load(ROOT + f'lane_cv_20261108/out/tta_peaks/{tag}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
        for k, i in enumerate(ix): out[int(i)] = pk[of[k]:of[k + 1]]
    return out


PK = {t: load(t) for t in ('seeds3_tta4', 's46_tta4', 's47_tta4', 's48_tta4')}
z = np.load(ROOT + 'lane_cellroute_20261109/feats/seg_centroids.npz'); _pts, _of, _ix = z['pts'], z['offsets'], z['indices']; SEG = {int(d): _pts[_of[k]:_of[k + 1]] for k, d in enumerate(_ix)}
SETS = ('P2', 'P0'); REGS = ('box', 'p150', 'win', 'full'); TR = ('raw', 'trim')
COLS = [f'{s}_{r}_{t}_{c}' for s in SETS for r in REGS for t in TR for c in ('cv_k0', 'cv_k2', 'hex_k0', 'hex_k2', 'n')]


def st(res, m):
    if m.sum() < 10: return [np.nan] * 4 + [float(m.sum())]
    a0, s0, a2, s2 = res[0][0][m], res[0][1][m], res[2][0][m], res[2][1][m]
    return [a0.std() / a0.mean(), a2.std() / a2.mean(), float(np.mean(s0 == 6)), float(np.mean(s2 == 6)), float(m.sum())]


def one(r):
    out = [np.nan] * len(COLS); pk = PK['seeds3_tta4'].get(r)
    if pk is None or len(pk) < 20: return r, out
    pk = pk.astype(float); hi = pk[pk[:, 2] >= 0.5, :2]; mid = pk[pk[:, 2] >= 0.35, :2]; ref = hi if len(hi) > 20 else mid
    if len(ref) < 10: return r, out
    s = float(np.median(cKDTree(ref).query(ref, k=2)[0][:, 1])); wins = LW.windows(ref, 380, 300, k=3); x0, y0 = LW.windows(ref, BW, BH, k=1)[0]
    hb = ref[(ref[:, 0] >= x0) & (ref[:, 0] < x0 + BW) & (ref[:, 1] >= y0) & (ref[:, 1] < y0 + BH)]; c0 = hb.mean(0) if len(hb) else np.array([x0 + BW / 2, y0 + BH / 2])
    sg = SEG.get(r, np.zeros((0, 3))).astype(float); sg = sg[sg[:, 2] >= 0.15 * s ** 2]
    srcs = [PK[t][r].astype(float) for t in ('s46_tta4', 's47_tta4', 's48_tta4')]; srcs = [x[x[:, 2] >= 0.35] for x in srcs] + [np.c_[sg[:, :2], np.full(len(sg), 0.5)]]
    P2 = L.vote(srcs, s, 0.45, 1)[0]; q = pk[pk[:, 2] >= 0.2, :2]; vals = []
    for p in (P2, q):
        ix, res, nb = L.konan_cells_idx(p); pts = p[ix] if len(ix) else np.zeros((0, 2))
        box = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + BW) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + BH)
        o = np.where(box)[0]; o = o[np.argsort(np.linalg.norm(pts[o] - c0, axis=1))][:150]; p150 = np.zeros(len(pts), bool); p150[o] = True
        win = np.zeros(len(pts), bool)
        for a, b in wins: win |= (pts[:, 0] >= a) & (pts[:, 0] < a + 380) & (pts[:, 1] >= b) & (pts[:, 1] < b + 300)
        for m in (box, p150, win, np.ones(len(pts), bool)):
            vals += st(res, m)
            if m.sum() >= 10:
                med = np.median(res[2][0][m]); mt = m & (res[2][0] > 0.4 * med) & (res[2][0] < 2.5 * med)
            else: mt = m
            vals += st(res, mt)
    return r, vals


if __name__ == '__main__':
    W_ = int(sys.argv[1]) if len(sys.argv) > 1 else 4; F = np.full((N, len(COLS)), np.nan); t0 = time.time(); done = 0
    with get_context('fork').Pool(W_) as pool:
        for r, v in pool.imap_unordered(one, range(N), chunksize=8):
            F[r] = v; done += 1
            if done % 1000 == 0: print(done, 'frames', round((time.time() - t0) / 60, 1), 'min', flush=True)
    np.savez_compressed(ROOT + 'lane_cellroute_20261109/feats/trim_feats.npz', F=F, names=np.array(COLS), note=__doc__)
    print('done', int(np.isfinite(F[:, 1]).sum()), 'frames', flush=True)
