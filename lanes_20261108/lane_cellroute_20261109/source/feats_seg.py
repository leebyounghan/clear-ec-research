"""Cell-route lane (2026-11-09): label-free cell statistics from the zero-shot Cellpose-SAM loose masks only (no detector, no CLEAR-EC-trained
weights anywhere): seg centroids -> spacing s (median NN of centroids with area >= 30 px), points = centroids with area >= 0.15 s^2;
Konan k=0/k=2 cells over the whole frame (interior seeds); regions win (3 legible 380x300 windows by point count), full, box (538x408);
raw and trimmed (cell area within 0.4-2.5 x region median). Columns per region/trim: cv_k0, cv_k2, hex_k0, hex_k2, n, cd_k0 (1e6 / mean k0 area um^2).
usage: feats_seg.py WORKERS -> feats/seg_konan_feats.npz (dataset index order)."""
import os, sys, time
from multiprocessing import get_context
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source')
import cr_lib as L
import deploy_hexkonan as D
LANE = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/'; N = 8954; UM2 = 0.7716 ** 2
z = np.load(LANE + 'feats/seg_centroids.npz'); _p, _o, _i = z['pts'], z['offsets'], z['indices']; SEG = {int(d): _p[_o[k]:_o[k + 1]] for k, d in enumerate(_i)}
COLS = ['s_seg'] + [f'seg_{r}_{t}_{c}' for r in ('win', 'full', 'box') for t in ('raw', 'trim') for c in ('cv_k0', 'cv_k2', 'hex_k0', 'hex_k2', 'n', 'cd_k0')]


def st(res, m):
    if m.sum() < 10: return [np.nan] * 4 + [float(m.sum()), np.nan]
    a0, s0, a2, s2 = res[0][0][m], res[0][1][m], res[2][0][m], res[2][1][m]
    return [a0.std() / a0.mean(), a2.std() / a2.mean(), float(np.mean(s0 == 6)), float(np.mean(s2 == 6)), float(m.sum()), 1e6 / (a0.mean() * UM2)]


def one(r):
    out = [np.nan] * len(COLS); sg = SEG.get(r)
    if sg is None or len(sg) < 30: return r, out
    sg = sg.astype(float); big = sg[sg[:, 2] >= 30, :2]
    if len(big) < 20: return r, out
    s = float(np.median(cKDTree(big).query(big, k=2)[0][:, 1])); p = sg[sg[:, 2] >= 0.15 * s ** 2, :2]
    if len(p) < 20: return r, out
    wins = D.windows(p, 380, 300, k=3); bx, by = D.windows(p, 538, 408, k=1)[0]
    ix, res, nb = L.konan_cells_idx(p); pts = p[ix] if len(ix) else np.zeros((0, 2)); vals = [s]
    win = np.zeros(len(pts), bool)
    for a, b in wins: win |= (pts[:, 0] >= a) & (pts[:, 0] < a + 380) & (pts[:, 1] >= b) & (pts[:, 1] < b + 300)
    box = (pts[:, 0] >= bx) & (pts[:, 0] < bx + 538) & (pts[:, 1] >= by) & (pts[:, 1] < by + 408)
    for m in (win, np.ones(len(pts), bool), box):
        vals += st(res, m)
        if m.sum() >= 10:
            med = np.median(res[2][0][m]); mt = m & (res[2][0] > 0.4 * med) & (res[2][0] < 2.5 * med)
        else: mt = m
        vals += st(res, mt)
    return r, vals


if __name__ == '__main__':
    W_ = int(sys.argv[1]) if len(sys.argv) > 1 else 16; F = np.full((N, len(COLS)), np.nan); t0 = time.time(); done = 0
    with get_context('fork').Pool(W_) as pool:
        for r, v in pool.imap_unordered(one, range(N), chunksize=8):
            F[r] = v; done += 1
            if done % 2000 == 0: print(done, 'frames', round((time.time() - t0) / 60, 1), 'min', flush=True)
    np.savez_compressed(LANE + 'feats/seg_konan_feats.npz', F=F, names=np.array(COLS), note=__doc__)
    print('done', int(np.isfinite(F[:, 2]).sum()), 'frames', flush=True)
