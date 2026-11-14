"""Distribution of noise, blur and exposure across labelled frames, calibration of the estimators, and link to labels and best-pipeline residuals (lane GPU 0, 2026-11-06)."""
import json, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, 'experiments/lane_gpu0_20261106/source')
import measure_frames_20261106 as M

X = Path('experiments')
LANE = X / 'lane_gpu0_20261106'
z = np.load(LANE / 'frame_stats.npz', allow_pickle=True)
ids, y, names, S, LP, freq = z['ids'].astype(str), z['y'], list(z['names'].astype(str)), z['stats'].astype(np.float64), z['log_power'].astype(np.float64), z['freq']
col = {n: i for i, n in enumerate(names)}
BAND = (0.08, 0.25)


def blur2(lp):
    """-slope of log(P - floor) vs f^2 over BAND, divided by 4 pi^2: the Gaussian sigma^2 of the spectrum's roll-off. Blur adds sigma^2 directly."""
    P = np.exp(lp)
    floor = P[..., freq > 0.42].mean(-1, keepdims=True)
    S_ = np.maximum(P - floor, 1e-3 * floor)
    m = (freq >= BAND[0]) & (freq <= BAND[1])
    xx = freq[m] ** 2
    xc = xx - xx.mean()
    yy = np.log(S_[..., m])
    slope = (yy * xc).sum(-1) / (xc ** 2).sum()
    return -slope / (4 * np.pi ** 2)


def source_year(i):
    i = i.strip()
    m = re.fullmatch(r'\d+-(\d\d)', i)
    if m:
        return 'A', 2000 + int(m.group(1))
    m = re.match(r'(\d{4})-', i)
    return ('B', int(m.group(1))) if m else ('?', 0)


src = np.array([source_year(i)[0] for i in ids]); yr = np.array([source_year(i)[1] for i in ids])
bb = blur2(LP)
F = {n: S[:, col[n]] for n in names}
F['blur2'] = bb
F['floor_log'] = LP[:, freq > 0.42].mean(1)
F['snr_hp'] = F['hp_std'] / (F['noise_imm'] + 1e-6)
F['contrast'] = F['hp_std'] / (F['mean'] + 1e-6)
F['p99_over_p01'] = F['p99'] / (F['p01'] + 1)
F['gamma_proxy'] = np.log((F['p50'] + 1) / (F['p95'] + 1)) / np.log(0.5)  # log(median/p95) against a linear ramp
KEY = ['mean', 'p01', 'p50', 'p99', 'p999', 'frac255', 'frac0', 'levels', 'center_over_corner', 'illum_cv', 'grad_lr', 'grad_tb', 'hp_std', 'contrast', 'noise_imm', 'noise_imm_dark', 'noise_imm_mid', 'noise_imm_bright', 'res_ac1h', 'res_ac1v', 'res_ac2h', 'lap_var', 'blur2', 'floor_log', 'snr_hp', 'gamma_proxy', 'blk8h', 'blk8v', 'interlace', 'row_std', 'col_std', 'axis_frac_hf']
Q = [1, 5, 25, 50, 75, 95, 99]


def table(mask, label):
    print(f'\n## {label} n={int(mask.sum())}')
    print('feature'.ljust(20), *[f'p{q}'.rjust(8) for q in Q])
    for k in KEY:
        print(k.ljust(20), *[f'{v:8.3f}' for v in np.percentile(F[k][mask], Q)])


table(np.ones(len(ids), bool), 'all')
table(src == 'A', 'source A')
table(src == 'B', 'source B')
for yy in sorted(set(yr[src == 'A'])):
    print(f'A {yy} n={(yr[(src == "A")] == yy).sum()}', {k: round(float(np.median(F[k][(src == "A") & (yr == yy)])), 3) for k in ('mean', 'contrast', 'noise_imm', 'blur2', 'center_over_corner', 'floor_log', 'levels')})
for yy in sorted(set(yr[src == 'B'])):
    print(f'B {yy} n={(yr[(src == "B")] == yy).sum()}', {k: round(float(np.median(F[k][(src == "B") & (yr == yy)])), 3) for k in ('mean', 'contrast', 'noise_imm', 'blur2', 'center_over_corner', 'floor_log', 'levels')})
print('\nmedian log power by frequency, A then B:')
for a, b_, c in zip(freq, np.median(LP[src == 'A'], 0), np.median(LP[src == 'B'], 0)):
    print(f'{a:.3f} {b_:7.2f} {c:7.2f}')


# Calibration: known blur and noise on 120 random frames.
def calib(i):
    cv2.setNumThreads(1)
    raw = np.memmap(M.RAW, mode='r', dtype=np.uint8, shape=(M.TOTAL, M.H, M.W))
    a = np.asarray(raw[i]); rng = np.random.default_rng(int(i)); out = {}
    out['none'] = M.frame_stats(a)
    for s in (0.5, 1.0, 1.5, 2.0):
        out[f'blur{s}'] = M.frame_stats(np.clip(np.rint(cv2.GaussianBlur(a.astype(np.float32), (0, 0), s)), 0, 255).astype(np.uint8))
    for s in (2.0, 4.0, 6.0):
        out[f'noise{s}'] = M.frame_stats(np.clip(np.rint(a.astype(np.float32) + rng.normal(0, s, a.shape)), 0, 255).astype(np.uint8))
    return out


if __name__ == '__main__':
    pick = np.random.default_rng(1).choice(len(ids), 120, replace=False)
    with Pool(4) as p:
        R = p.map(calib, pick.tolist(), chunksize=4)
    print('\n## calibration, 120 frames: median of (variant minus none) for blur2; noise_imm median')
    base_blur = blur2(np.stack([r['none'][1].astype(np.float64) for r in R]))
    base_noise = np.array([r['none'][0][col['noise_imm']] for r in R])
    base_lap = np.array([r['none'][0][col['lap_var']] for r in R])
    for v in R[0]:
        b2 = blur2(np.stack([r[v][1].astype(np.float64) for r in R]))
        nn = np.array([r[v][0][col['noise_imm']] for r in R])
        lp = np.array([r[v][0][col['lap_var']] for r in R])
        print(v.ljust(10), 'd blur2 median', round(float(np.median(b2 - base_blur)), 3), 'IQR', np.round(np.percentile(b2 - base_blur, [25, 75]), 3).tolist(), 'noise_imm median', round(float(np.median(nn)), 3), 'expected sqrt(n0^2+s^2) uses n0', round(float(np.median(base_noise)), 3), 'lap ratio', round(float(np.median(lp / base_lap)), 3))
    sys.path.insert(0, str(X / 'meta_ceiling_20261103/source'))
    import best_pipeline_20261106 as BP
    rows = BP.idx
    assert all(re.sub(r'\s+', ' ', str(a).strip()) == re.sub(r'\s+', ' ', str(b_).strip()) for a, b_ in zip(ids[rows], BP.ids))
    res = np.log(BP.REF_BEST) - np.log(np.maximum(BP.y, 1e-6))
    ok = BP.y.min(1) > 0
    print('\n## Spearman with label and with best-pipeline log residual (prediction / label), within source. n rows', len(rows))
    print('feature'.ljust(20), 'A:lab CD CV HEX | res CD CV HEX | absres CD CV HEX || B: same')
    for k in KEY:
        line = []
        for s_ in ('A', 'B'):
            m = (src[rows] == s_) & ok
            v = F[k][rows][m]
            a = [spearmanr(v, BP.y[m, j]).statistic for j in range(3)]
            b_ = [spearmanr(v, res[m, j]).statistic for j in range(3)]
            c = [spearmanr(v, np.abs(res[m, j])).statistic for j in range(3)]
            line.append(' '.join(f'{u:+.2f}' for u in a) + ' | ' + ' '.join(f'{u:+.2f}' for u in b_) + ' | ' + ' '.join(f'{u:+.2f}' for u in c))
        print(k.ljust(20), ' || '.join(line))
    np.savez_compressed(LANE / 'frame_features_derived.npz', ids=ids, src=src, yr=yr, **{k: F[k].astype(np.float32) for k in KEY})
