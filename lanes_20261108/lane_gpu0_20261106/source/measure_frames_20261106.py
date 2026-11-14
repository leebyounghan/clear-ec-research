"""Per-frame noise, blur and exposure on the raw frames, before CLAHE, all labelled frames (lane GPU 0, 2026-11-06).
Label-free. Writes frame_stats.npz in the lane folder. Four worker processes, one thread each."""
import sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

H, W = 972, 1296
LANE = Path('experiments/lane_gpu0_20261106')
RAW = 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat'
META = 'datasets/clear_ec_official/cache_native_meta_512.npz'
NBIN = 48
N_IMM = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
# Radial bins for the central square
_S = 972
_fy = np.fft.fftfreq(_S)[:, None]
_fx = np.fft.rfftfreq(_S)[None, :]
_R = np.sqrt(_fy ** 2 + _fx ** 2)
_EDGES = np.linspace(0, 0.5, NBIN + 1)
_BIN = np.clip(np.digitize(_R.ravel(), _EDGES) - 1, 0, NBIN)
_COUNT = np.bincount(_BIN, minlength=NBIN + 1)[:NBIN].astype(np.float64)
_WIN = np.outer(np.hanning(_S), np.hanning(_S)).astype(np.float32)
_AXMASK = (np.abs(_fx) < 0.01) | (np.abs(_fy) < 0.01)  # axis-aligned energy (line noise, block grid)
FREQ = 0.5 * (_EDGES[1:] + _EDGES[:-1])
NAMES = ['mean', 'p001', 'p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99', 'p999', 'frac0', 'frac255', 'levels', 'black_rows', 'black_cols',
         'center_over_corner', 'illum_cv', 'grad_lr', 'grad_tb', 'hp_std', 'hp_p01', 'hp_p99', 'hp_skew',
         'noise_imm', 'noise_imm_dark', 'noise_imm_mid', 'noise_imm_bright', 'mean_dark', 'mean_mid', 'mean_bright',
         'res_ac1h', 'res_ac1v', 'res_ac2h', 'res_ac2v', 'lap_var', 'tenengrad', 'row_std', 'col_std', 'interlace', 'blk8h', 'blk8v', 'axis_frac_hf']


def frame_stats(a):
    f = a.astype(np.float32)
    q = np.percentile(a, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    hist = np.bincount(a.ravel(), minlength=256)
    low = cv2.GaussianBlur(f, (0, 0), 40)
    c = low[H // 3:2 * H // 3, W // 3:2 * W // 3].mean()
    k = H // 6
    corners = np.mean([low[:k, :k].mean(), low[:k, -k:].mean(), low[-k:, :k].mean(), low[-k:, -k:].mean()])
    mean = f.mean()
    grad_lr = (low[:, W // 2:].mean() - low[:, :W // 2].mean()) / (mean + 1e-6)
    grad_tb = (low[H // 2:].mean() - low[:H // 2].mean()) / (mean + 1e-6)
    hp = f - cv2.GaussianBlur(f, (0, 0), 15)
    hps = hp.std()
    hq = np.percentile(hp, [1, 99])
    conv = cv2.filter2D(f, cv2.CV_32F, N_IMM, borderType=cv2.BORDER_REFLECT)[2:-2, 2:-2]
    ac = np.abs(conv)
    imm = np.median(ac) / 0.6745 / 6
    lp = cv2.GaussianBlur(f, (0, 0), 3)[2:-2, 2:-2]
    t1, t2 = np.percentile(lp[::4, ::4], [33.3, 66.7])
    masks = (lp <= t1, (lp > t1) & (lp <= t2), lp > t2)
    imms = [np.median(ac[m]) / 0.6745 / 6 for m in masks]
    means = [lp[m].mean() for m in masks]
    # Residual correlation: from a fine high pass over the flattest half of the frame.
    r = f - cv2.GaussianBlur(f, (0, 0), 1.5)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    gm = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), 2)
    flat = gm <= np.median(gm)

    def corr(dy, dx):
        A, B = r[:H - dy, :W - dx], r[dy:, dx:]
        M = flat[:H - dy, :W - dx] & flat[dy:, dx:]
        x, y = A[M], B[M]
        return float(np.dot(x, y) / (np.sqrt(np.dot(x, x) * np.dot(y, y)) + 1e-6))
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=1)
    ten = float(np.mean(gx * gx + gy * gy))
    rows, cols = f.mean(1), f.mean(0)
    row_hp = rows - cv2.GaussianBlur(rows[None], (0, 0), 3).ravel()
    col_hp = cols - cv2.GaussianBlur(cols[None], (0, 0), 3).ravel()
    odd, even = f[1::2], f[0::2]
    n = min(len(odd), len(even))
    interlace = float(np.abs(odd[:n] - even[:n]).mean())
    dh, dv = np.abs(np.diff(f, axis=1)).mean(0), np.abs(np.diff(f, axis=0)).mean(1)
    bh = np.arange(len(dh)) % 8 == 7
    bv = np.arange(len(dv)) % 8 == 7
    x = f[:_S, (W - _S) // 2:(W - _S) // 2 + _S]
    x = (x - x.mean()) * _WIN
    P = np.abs(np.fft.rfft2(x)) ** 2
    spec = np.bincount(_BIN, weights=P.ravel(), minlength=NBIN + 1)[:NBIN] / _COUNT
    hf = _R > 0.3
    axis_frac = P[hf & _AXMASK].sum() / (P[hf].sum() + 1e-6)
    v = [mean, *q, hist[0] / a.size, hist[255] / a.size, (hist > 0).sum(), float((f.max(1) == 0).mean()), float((f.max(0) == 0).mean()),
         c / (corners + 1e-6), low.std() / (mean + 1e-6), grad_lr, grad_tb, hps, hq[0], hq[1], float(((hp / (hps + 1e-6)) ** 3).mean()),
         imm, *imms, *means, corr(0, 1), corr(1, 0), corr(0, 2), corr(2, 0), float(lap.var()), ten, row_hp.std(), col_hp.std(), interlace,
         dh[bh].mean() / (dh[~bh].mean() + 1e-6), dv[bv].mean() / (dv[~bv].mean() + 1e-6), axis_frac]
    return np.array(v, np.float32), np.log(spec + 1e-3).astype(np.float32)


_raw = None


def work(i):
    global _raw
    cv2.setNumThreads(1)
    if _raw is None:
        _raw = np.memmap(RAW, mode='r', dtype=np.uint8, shape=(TOTAL, H, W))
    return frame_stats(np.asarray(_raw[i]))


if __name__ == '__main__':
    with np.load(META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float64)
    TOTAL = len(ids)
    t = time.time()
    with Pool(4, initializer=lambda: None) as pool:
        out = []
        for j, r in enumerate(pool.imap(work, range(TOTAL), chunksize=32)):
            out.append(r)
            if j % 1000 == 0:
                print(j, round(time.time() - t), flush=True)
    S = np.stack([a for a, _ in out])
    P = np.stack([b for _, b in out])
    np.savez_compressed(LANE / 'frame_stats.npz', ids=ids, y=y, names=np.array(NAMES), stats=S, log_power=P, freq=FREQ)
    print('saved', S.shape, P.shape, round(time.time() - t), flush=True)
else:
    with np.load(META, allow_pickle=True) as z:
        TOTAL = len(z['ids'])
