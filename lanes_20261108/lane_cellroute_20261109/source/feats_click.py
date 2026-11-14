"""Cell-route lane (2026-11-09): click-noise emulation of the device label process on all 8954 frames.
The device computes CV/HEX from operator clicks; clicks scatter ~0.16 spacing around the machine consensus (diag_click.py, measured on the
25 marker frames, not tuned). Feature = mean over K draws of the Konan k=0/k=2 statistics of (points + N(0, 0.16 s)), cells whose seeds lie in the
3 legible 380x300 windows (Delaunay over the window points plus a 3 s margin). Point sets: P2 = vote of s46/s47/s48 TTA peaks >= 0.35 and
seg-mask centroids (>= 1 source); optional P5 = vote of s46-s50 + seg (MINV5 sources). Columns per set: cv_k0, cv_k2, hex_k0, hex_k2, n, s5_k2, s7_k2.
usage: feats_click.py WORKERS K SETS(comma: P2,P5) [MINV5]; writes feats/click_feats_<SETS>.npz (dataset index order)."""
import os, sys, time
from multiprocessing import get_context
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from scipy.spatial import cKDTree
sys.path[:0] = ['/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source', '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source']
import cr_lib as L
import legible_window_features_20261105 as LW
ROOT = '/workspace/clear-ec-research/experiments/'; N = 8954; SIG = 0.16
W_ = int(sys.argv[1]); K = int(sys.argv[2]); SETS = sys.argv[3].split(','); MINV5 = int(sys.argv[4]) if len(sys.argv) > 4 else 1
TAGS = ['seeds3_tta4', 's46_tta4', 's47_tta4', 's48_tta4'] + (['s49_tta4', 's50_tta4'] if 'P5' in SETS else [])


def load(tag):
    out = {}
    for f in range(5):
        z = np.load(ROOT + f'lane_cv_20261108/out/tta_peaks/{tag}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ix = z['peaks'], z['offsets'], z['indices']
        for k, i in enumerate(ix): out[int(i)] = pk[of[k]:of[k + 1]]
    return out


PK = {t: load(t) for t in TAGS}
z = np.load(ROOT + 'lane_cellroute_20261109/feats/seg_centroids.npz'); _pts, _of, _ix = z['pts'], z['offsets'], z['indices']; SEG = {int(d): _pts[_of[k]:_of[k + 1]] for k, d in enumerate(_ix)}
COLS = [f'{s}c_win_{c}' for s in SETS for c in ('cv_k0', 'cv_k2', 'hex_k0', 'hex_k2', 'n', 's5_k2', 's7_k2')]


def one(r):
    out = [np.nan] * len(COLS); pk = PK['seeds3_tta4'].get(r)
    if pk is None or len(pk) < 20: return r, out
    pk = pk.astype(float); hi = pk[pk[:, 2] >= 0.5, :2]; mid = pk[pk[:, 2] >= 0.35, :2]; ref = hi if len(hi) > 20 else mid
    if len(ref) < 10: return r, out
    s = float(np.median(cKDTree(ref).query(ref, k=2)[0][:, 1])); wins = LW.windows(ref, 380, 300, k=3)
    sg = SEG.get(r, np.zeros((0, 3))).astype(float); sg = sg[sg[:, 2] >= 0.15 * s ** 2]; sgp = np.c_[sg[:, :2], np.full(len(sg), 0.5)]
    rng = np.random.default_rng(r); vals = []
    for nm in SETS:
        dets = ['s46_tta4', 's47_tta4', 's48_tta4'] + (['s49_tta4', 's50_tta4'] if nm == 'P5' else [])
        srcs = [PK[t][r].astype(float) for t in dets]; srcs = [x[x[:, 2] >= 0.35] for x in srcs] + [sgp]
        p, w = L.vote(srcs, s, 0.45, 1 if nm == 'P2' else MINV5)
        inw = np.zeros(len(p), bool); near = np.zeros(len(p), bool); m3 = 3 * s
        for x0, y0 in wins:
            inw |= (p[:, 0] >= x0) & (p[:, 0] < x0 + 380) & (p[:, 1] >= y0) & (p[:, 1] < y0 + 300)
            near |= (p[:, 0] >= x0 - m3) & (p[:, 0] < x0 + 380 + m3) & (p[:, 1] >= y0 - m3) & (p[:, 1] < y0 + 300 + m3)
        q = p[near]; sel = inw[near]; acc = []
        for _ in range(K):
            res = L.konan_cells(q + rng.normal(0, SIG * s, q.shape), sel)
            a0, s0 = np.asarray(res[0][0]), np.asarray(res[0][1]); a2, s2 = np.asarray(res[2][0]), np.asarray(res[2][1])
            if len(a2) < 10 or len(a0) < 10: continue
            acc.append([a0.std() / a0.mean(), a2.std() / a2.mean(), np.mean(s0 == 6), np.mean(s2 == 6), len(a2), np.mean(s2 == 5), np.mean(s2 == 7)])
        vals += list(np.mean(np.array(acc), 0)) if acc else [np.nan] * 7
    return r, vals


if __name__ == '__main__':
    F = np.full((N, len(COLS)), np.nan); t0 = time.time(); done = 0
    with get_context('fork').Pool(W_) as pool:
        for r, v in pool.imap_unordered(one, range(N), chunksize=8):
            F[r] = v; done += 1
            if done % 500 == 0: print(done, 'frames', round((time.time() - t0) / 60, 1), 'min', flush=True)
    tag = '_'.join(SETS) + (f'_m{MINV5}' if 'P5' in SETS else '') + f'_K{K}'
    np.savez_compressed(ROOT + f'lane_cellroute_20261109/feats/click_feats_{tag}.npz', F=F, names=np.array(COLS), note=__doc__)
    print('done', int(np.isfinite(F[:, 3]).sum()), 'frames', flush=True)
