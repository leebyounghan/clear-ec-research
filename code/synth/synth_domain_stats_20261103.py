"""Appearance statistics to tune the synthetic renderer against real frames (2026-11-03). Label-free apart from the
expected cell spacing, which comes from CD. Per image, on the raw frame and on the CLAHE + 1/99 percentile input:
  raw_mean, raw_std, black_frac         global brightness and vignette
  lap_sd                                 high-frequency energy of the CLAHE input
  hp_contrast_p10/p50/p90                tile contrast after a sigma-10 high-pass (how much of the frame has texture)
  ring_frac                              share of tissue tiles whose radial spectrum has a peak at the cell frequency
  ring_height_p50                        median height of that peak (cell legibility)
usage: synth_domain_stats_20261103.py real N OUT.json | synth DIR N OUT.json. Forked pool, clamp(16 - load1 - 1, 2, 12).
"""
import json, os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
from scipy import ndimage as ndi
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
cv2.setNumThreads(0)
TILE = 128
CL = cv2.createCLAHE(2.0, (8, 8))


def spacing_px(cd):
    return np.sqrt(2 * (1e6 / cd) / np.sqrt(3)) / 0.77


def stats(a, cd):
    u = CL.apply(a)
    lo, hi = np.percentile(u, (1, 99))
    x = np.clip((u.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    hp = x - ndi.gaussian_filter(x, 10)
    H, W = x.shape
    con, ring = [], []
    f0 = 1 / spacing_px(cd)
    win = np.outer(np.hanning(TILE), np.hanning(TILE))
    fy, fx = np.meshgrid(np.fft.fftfreq(TILE), np.fft.fftfreq(TILE), indexing='ij')
    fr = np.hypot(fx, fy)
    band = (fr > 0.75 * f0) & (fr < 1.3 * f0)
    side = ((fr > 0.45 * f0) & (fr < 0.7 * f0)) | ((fr > 1.4 * f0) & (fr < 1.9 * f0))
    for y0 in range(0, H - TILE + 1, TILE):
        for x0 in range(0, W - TILE + 1, TILE):
            if a[y0:y0 + TILE, x0:x0 + TILE].mean() < 5:
                continue
            t = hp[y0:y0 + TILE, x0:x0 + TILE]
            con.append(t.std())
            P = np.abs(np.fft.fft2((t - t.mean()) * win)) ** 2
            ring.append(P[band].mean() / (P[side].mean() + 1e-9))
    con, ring = np.array(con), np.array(ring)
    lap = cv2.Laplacian(x, cv2.CV_32F)
    return dict(raw_mean=float(a.mean()), raw_std=float(a.std()), black_frac=float((a[::2, ::2] < 5).mean()), lap_sd=float(lap.std()),
                hp_contrast_p10=float(np.percentile(con, 10)), hp_contrast_p50=float(np.percentile(con, 50)), hp_contrast_p90=float(np.percentile(con, 90)),
                ring_frac=float((ring > 1.5).mean()), ring_height_p50=float(np.median(ring)))


def real_task(r):
    i, cd = r
    mm = np.memmap('/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
    return stats(np.asarray(mm[i]), cd)


def synth_task(r):
    path, cd = r
    from PIL import Image
    return stats(np.asarray(Image.open(path)), cd)


def main():
    mode = sys.argv[1]
    workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1)))
    if mode == 'real':
        n, out = int(sys.argv[2]), sys.argv[3]
        sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
        import eval_oof_20261031 as E
        b = E.load_base(); ok = np.where(b['y'][:, 0] > 0)[0]
        rows = np.random.default_rng(0).choice(ok, n, replace=False)
        tasks = [(int(b['idx'][r]), float(b['y'][r, 0])) for r in rows]; fn = real_task
    else:
        d, n, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        names = sorted(os.listdir(f'{d}/meta'))[:n]
        tasks = []
        for name in names:
            with np.load(f'{d}/meta/{name}') as z:
                tasks.append((f'{d}/images/{name[:-4]}.png', float(z['target'][0])))
        fn = synth_task
    with get_context('fork').Pool(workers) as pool:
        res = list(pool.imap(fn, tasks, chunksize=4))
    keys = list(res[0])
    summary = {k: [round(float(np.percentile([r[k] for r in res], q)), 4) for q in (10, 50, 90)] for k in keys}
    json.dump(dict(n=len(res), summary=summary, rows=res), open(out, 'w'))
    print(json.dumps(dict(mode=mode, n=len(res), p10_p50_p90=summary)), flush=True)


if __name__ == '__main__':
    main()
