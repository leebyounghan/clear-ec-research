"""Second calibration: focus estimator that survives the noise floor, and noise sigma from the white floor (lane GPU 0, 2026-11-06)."""
import json, re, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, 'experiments/lane_gpu0_20261106/source')
import measure_frames_20261106 as M

LANE = Path('experiments/lane_gpu0_20261106')
z = np.load(LANE / 'frame_stats.npz', allow_pickle=True)
ids, names, S, LP, freq = z['ids'].astype(str), list(z['names'].astype(str)), z['stats'].astype(np.float64), z['log_power'].astype(np.float64), z['freq']
SW2 = (np.hanning(972) ** 2).sum() ** 2
FLOOR = (freq >= 0.27) & (freq <= 0.40)


def sigma_floor(lp):
    return np.sqrt(np.exp(lp[..., FLOOR]).mean(-1) / SW2)


def focus2(lp, band):
    P = np.exp(lp)
    fl = P[..., FLOOR].mean(-1, keepdims=True)
    S_ = np.maximum(P - fl, 1e-2 * fl)
    m = (freq >= band[0]) & (freq <= band[1])
    x = freq[m] ** 2
    xc = x - x.mean()
    return -(np.log(S_[..., m]) * xc).sum(-1) / (xc ** 2).sum() / (4 * np.pi ** 2)


def refill(a, s, rng):
    """Defocus: blur the frame, then put back the white noise that the blur removed."""
    n0 = float(np.sqrt(np.exp(np.log(1e-9 + 1)) * 0))
    return a


def calib(i):
    cv2.setNumThreads(1)
    raw = np.memmap(M.RAW, mode='r', dtype=np.uint8, shape=(M.TOTAL, M.H, M.W))
    a = np.asarray(raw[i]); rng = np.random.default_rng(int(i)); out = {}
    _, lp0 = M.frame_stats(a); out['none'] = lp0
    s0 = float(sigma_floor(lp0.astype(np.float64)))
    for s in (0.5, 0.75, 1.0, 1.5, 2.0):
        b = cv2.GaussianBlur(a.astype(np.float32), (0, 0), s)
        out[f'blur{s}'] = M.frame_stats(np.clip(np.rint(b), 0, 255).astype(np.uint8))[1]
        k = cv2.getGaussianKernel(int(2 * np.ceil(4 * s) + 1), s)
        att = float((k ** 2).sum())
        fill = s0 * np.sqrt(max(0.0, 1 - att ** 2))
        out[f'defocus{s}'] = M.frame_stats(np.clip(np.rint(b + rng.standard_normal(a.shape, dtype=np.float32) * fill), 0, 255).astype(np.uint8))[1]
    for s in (2.0, 4.0, 6.0):
        out[f'noise{s}'] = M.frame_stats(np.clip(np.rint(a.astype(np.float32) + rng.standard_normal(a.shape, dtype=np.float32) * s), 0, 255).astype(np.uint8))[1]
    return out


if __name__ == '__main__':
    pick = np.random.default_rng(2).choice(len(ids), 120, replace=False)
    with Pool(4) as p:
        R = p.map(calib, pick.tolist(), chunksize=4)
    base = np.stack([r['none'] for r in R]).astype(np.float64)
    s0 = sigma_floor(base)
    print('sigma floor base median', np.median(s0).round(3))
    for v in R[0]:
        lp = np.stack([r[v] for r in R]).astype(np.float64)
        line = [v.ljust(12), 'sigma_floor', round(float(np.median(sigma_floor(lp))), 3)]
        if v.startswith('noise'):
            ss = float(v[5:])
            line += ['expected', round(float(np.median(np.sqrt(s0 ** 2 + ss ** 2))), 3)]
        for band in ((0.06, 0.12), (0.07, 0.14), (0.08, 0.16), (0.08, 0.20)):
            d = focus2(lp, band) - focus2(base, band)
            line += [band, 'd', round(float(np.median(d)), 3), 'IQR', np.round(np.percentile(d, [25, 75]), 2).tolist()]
        print(*line)
    print('\nreal distribution over all frames')
    src = np.array(['A' if re.fullmatch(r'\d+-\d\d', i.strip()) else 'B' for i in ids])
    Q = [1, 5, 25, 50, 75, 95, 99]
    for nm, v in (('sigma_floor', sigma_floor(LP)), ('focus2(.06,.12)', focus2(LP, (0.06, 0.12))), ('focus2(.07,.14)', focus2(LP, (0.07, 0.14))), ('focus2(.08,.16)', focus2(LP, (0.08, 0.16)))):
        print(nm.ljust(18), 'all', np.percentile(v, Q).round(2).tolist(), 'A', np.percentile(v[src == 'A'], Q).round(2).tolist(), 'B', np.percentile(v[src == 'B'], Q).round(2).tolist())
    y = z['y']
    from scipy.stats import spearmanr
    v = focus2(LP, (0.07, 0.14)); s = sigma_floor(LP)
    for sc in ('A', 'B'):
        m = (src == sc) & (y.min(1) > 0)
        print(sc, 'focus vs CD/CV/HEX', [round(float(spearmanr(v[m], y[m, j]).statistic), 3) for j in range(3)], 'sigma vs labels', [round(float(spearmanr(s[m], y[m, j]).statistic), 3) for j in range(3)], 'focus vs sigma', round(float(spearmanr(v[m], s[m]).statistic), 3))
