"""Robust cell-area statistics from the HEX lane's Cellpose-SAM masks (seg_loose, read-only) for CD (CD lane, 2026-11-09).
Per frame, interior cells = not touching the frame edge, area > 30 px, and less than 25 % of the 3-px ring around the cell is background (the HEX lane rule). Densities (cells/mm2, 0.7716 um/px):
mean area (as cd_mm2), median area, 10-90 % trimmed mean, mean over cells within 0.4-2.5 x median, and the mean area inside the
538 x 408 window (the device box size) holding the most interior-cell centroids (step 27 px). Also counts. 2 worker processes.
usage: seg_area_stats_20261109.py SEGDIR OUT.npz"""
import sys, glob, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy import ndimage as ndi
from multiprocessing import get_context
UM2 = 0.7716049 ** 2; BW, BH, STEP = 538, 408, 27
def frame_stats(M):
    M = M.astype(np.int64); n = int(M.max()); out = np.full(10, np.nan)
    if n < 10: return out
    area = np.bincount(M.ravel(), minlength=n + 1).astype(float)
    edge = np.unique(np.r_[M[0], M[-1], M[:, 0], M[:, -1]]); ok = area > 30; ok[0] = False; ok[edge] = False
    dil = ndi.grey_dilation(M, size=(7, 7)); ring = (M == 0) & (dil > 0)
    bg = np.bincount(dil[ring], minlength=n + 1).astype(float); per = np.bincount(dil[(dil > 0) & (dil != M)], minlength=n + 1).astype(float)
    ok &= (bg / np.maximum(per, 1)) < 0.25
    if ok.sum() < 10: return out
    H, W = M.shape; yy, xx = np.indices((H, W)); cy = np.bincount(M.ravel(), yy.ravel(), n + 1) / np.maximum(area, 1); cx = np.bincount(M.ravel(), xx.ravel(), n + 1) / np.maximum(area, 1)
    a = area[ok]; med = np.median(a); lo, hi = np.percentile(a, [10, 90]); tm = a[(a >= lo) & (a <= hi)].mean()
    band = a[(a > 0.4 * med) & (a < 2.5 * med)]
    ids = np.where(ok)[0]; best = (-1, None)
    for y0 in range(0, H - BH + 1, STEP):
        sy = (cy[ids] >= y0) & (cy[ids] < y0 + BH)
        for x0 in range(0, W - BW + 1, STEP):
            s = sy & (cx[ids] >= x0) & (cx[ids] < x0 + BW); c = int(s.sum())
            if c > best[0]: best = (c, s)
    wa = a[best[1]] if best[0] >= 10 else a
    d = lambda v: 1e6 / (v * UM2)
    return np.array([d(a.mean()), d(med), d(tm), d(band.mean()) if len(band) else np.nan, d(wa.mean()), d(np.median(wa)), float(ok.sum()), float(best[0]), float(len(band)), med])
def chunk(path):
    z = np.load(path); return z['frames'].astype(np.int64), np.stack([frame_stats(m) for m in z['masks']])
if __name__ == '__main__':
    paths = sorted(glob.glob(os.path.join(sys.argv[1], 'c*.npz'))); fr, st = [], []
    with get_context('fork').Pool(2) as pool:
        for k, (f, s) in enumerate(pool.imap(chunk, paths)):
            fr.append(f); st.append(s)
            if k % 20 == 0: print('chunk', k, len(paths), flush=True)
    np.savez_compressed(sys.argv[2], frames=np.concatenate(fr), S=np.concatenate(st), names=np.array(['cd_mean', 'cd_median', 'cd_trim', 'cd_band', 'cd_win_mean', 'cd_win_median', 'n_int', 'n_win', 'n_band', 'med_area_px']))
    print('done', flush=True)
