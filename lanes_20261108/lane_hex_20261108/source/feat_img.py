"""Richer whole-image statistics for the HEX residual (HEX lane, 2026-11-08). Deployable: computed from the raw frame only.
Row-ordered (base idx). Output feats/img_rich.npz with F (N x k) and names."""
import sys, numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L
idx = L.idx; N = len(idx)
mm = np.memmap('/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
H, W = 486, 648; yy, xx = np.mgrid[0:H, 0:W]; rad = np.hypot((yy - H / 2) / (H / 2), (xx - W / 2) / (W / 2))
cen = rad < 0.45; ring = rad > 0.95; Xd = np.c_[np.ones(H * W), (xx.ravel() - W / 2) / W, (yy.ravel() - H / 2) / H, rad.ravel() ** 2]
names = ['p05', 'p25', 'p50', 'p75', 'p95', 'mean', 'std', 'cen_mean', 'ring_mean', 'vig_ratio', 'gx', 'gy', 'r2', 'q_tl', 'q_tr', 'q_bl', 'q_br',
         'focus', 'focus_cen', 'focus_ring', 'sat_frac', 'black_frac', 'lstd_med', 'lstd_iqr', 'hf_ratio']
F = np.zeros((N, len(names)), np.float32)
proj = np.linalg.pinv(Xd.T @ Xd) @ Xd.T
for n, i in enumerate(idx):
    a = np.asarray(mm[i, ::2, ::2], dtype=np.float32); m = a.mean() + 1e-6
    p = np.percentile(a, [5, 25, 50, 75, 95]); cmean, rmean = a[cen].mean(), a[ring].mean(); c = proj @ a.ravel()
    lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]; lc, lr = cen[1:-1, 1:-1], ring[1:-1, 1:-1]
    b = a[:480, :640].reshape(30, 16, 40, 16).transpose(0, 2, 1, 3).reshape(30, 40, -1); ls = b.std(-1) / (b.mean(-1) + 1)
    sm = a[:480, :640].reshape(120, 4, 160, 4).mean((1, 3)); hf = np.mean(np.abs(a[:480, :640] - np.repeat(np.repeat(sm, 4, 0), 4, 1))) / m
    F[n] = [*(p / m), m, a.std() / m, cmean / m, rmean / m, rmean / (cmean + 1e-6), c[1] / m, c[2] / m, c[3] / m,
            a[:H // 2, :W // 2].mean() / m, a[:H // 2, W // 2:].mean() / m, a[H // 2:, :W // 2].mean() / m, a[H // 2:, W // 2:].mean() / m,
            np.log(lap.var() + 1), np.log(lap[lc].var() + 1), np.log(lap[lr].var() + 1), np.mean(a > 250), np.mean(a < 5),
            np.median(ls), np.subtract(*np.percentile(ls, [75, 25])), hf]
    if n % 1000 == 0: print(n, flush=True)
np.savez_compressed(f'{L.LANE}/feats/img_rich.npz', F=F, names=np.array(names), idx=idx); print('done', F.shape, flush=True)
