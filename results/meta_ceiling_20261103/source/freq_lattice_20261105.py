"""Reading the cell lattice in the frequency domain, without detecting a single cell (2026-11-05). Every point-based route hits the same wall: misses, false positives and jitter destroy Voronoi topology. A tile's power spectrum and autocorrelation need no points. For a tiling of cells:
  ring position    spacing, hence CD
  ring width       size disorder, physically tied to CV
  six-fold angular harmonic of the ring and of the autocorrelation shell: hexagonal order, tied to HEX
  autocorrelation second-peak height and decay: how far order persists
Each frame: CLAHE input, local contrast normalisation, 128 px tiles at 64 px stride, tissue tiles only. The ring frequency is found per frame, label-free. Per-tile values summarised over tiles (median, p25, p75, and the same on the top-third tiles by ring height, the legible region). Writes sweep/freq_lattice.npz. Forked pool, clamp(16 - load1 - 1, 2, 12), OMP 1."""
import os, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
cv2.setNumThreads(0)
ROOT = '/workspace/clear-ec-research'; RAW = f'{ROOT}/datasets/clear_ec_fullframe_20260904/img_full_raw.dat'; OUT = f'{ROOT}/experiments/meta_ceiling_20261103/sweep/freq_lattice.npz'
H, W, T, ST = 972, 1296, 128, 64
fy, fx = np.meshgrid(np.fft.fftfreq(T), np.fft.fftfreq(T), indexing='ij'); FR = np.hypot(fx, fy); TH = np.arctan2(fy, fx); WIN = np.outer(np.hanning(T), np.hanning(T))
BINS = np.linspace(0.015, 0.2, 38); RB = np.digitize(FR, BINS) - 1; MASK = (RB >= 0) & (RB < len(BINS) - 1); CENT = 0.5 * (BINS[1:] + BINS[:-1])
yy, xx = np.meshgrid(np.arange(T) - T // 2, np.arange(T) - T // 2, indexing='ij'); RR = np.hypot(yy, xx); AT = np.arctan2(yy, xx)
STAT = ['f0', 'height', 'width', 'psi6', 'psi4', 'psi2', 'ac_r1', 'ac_h1', 'ac_h2', 'ac_ratio', 'ac_psi6', 'contrast']
SUM = ['med', 'p25', 'p75', 'top_med']
NAMES = [f'{s}_{a}' for s in STAT for a in SUM] + ['tile_share', 'ring_share']


def one(i):
    a = np.asarray(np.memmap(RAW, dtype=np.uint8, mode='r', shape=(8954, H, W))[i]); u = cv2.createCLAHE(2.0, (W // 32, H // 32)).apply(a).astype(np.float32)
    bg = cv2.GaussianBlur(u, (0, 0), 24); sd = np.sqrt(cv2.GaussianBlur((u - bg) ** 2, (0, 0), 32)) + 2.0; n = (u - bg) / sd; tissue = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 8) > 12
    tiles = [n[y:y + T, x:x + T] for y in range(0, H - T + 1, ST) for x in range(0, W - T + 1, ST) if tissue[y:y + T, x:x + T].mean() > 0.9]
    if len(tiles) < 8:
        return i, np.full(len(NAMES), np.nan, np.float32)
    S = np.stack(tiles); S = (S - S.mean((1, 2), keepdims=True)) * WIN; P = np.abs(np.fft.fft2(S)) ** 2; nb = len(CENT)
    prof = np.stack([np.bincount(RB[MASK], p[MASK], nb) / np.maximum(np.bincount(RB[MASK], minlength=nb), 1) for p in P])
    # the frame's ring: peak of the mean whitened profile, label-free
    w = prof * CENT ** 2; k0 = int(np.argmax(np.mean(w / (w.mean(1, keepdims=True) + 1e-9), 0)[1:-1])) + 1; f_frame = CENT[k0]
    out = np.zeros((len(P), len(STAT)))
    for t, (p, g) in enumerate(zip(P, w)):
        lo, hi = max(0, k0 - 7), min(nb, k0 + 8); k = lo + int(np.argmax(g[lo:hi])); f0 = CENT[k]; base = np.median(g) + 1e-9; h = g[k] / base; half = g[k] / 2; a_ = k
        while a_ > 0 and g[a_] > half: a_ -= 1
        c_ = k
        while c_ < nb - 1 and g[c_] > half: c_ += 1
        wd = (CENT[c_] - CENT[a_]) / f0; ring = (FR > 0.8 * f0) & (FR < 1.25 * f0); m = p[ring]; th = TH[ring]; tot = m.sum() + 1e-9
        hs = [np.abs((m * np.exp(1j * q * th)).sum()) / tot for q in (6, 4, 2)]
        ac = np.fft.fftshift(np.fft.ifft2(p).real); ac = ac / (ac[T // 2, T // 2] + 1e-9); s_ = 1 / f0; rb = np.clip(np.round(RR).astype(int), 0, T // 2 + 40); rad = np.bincount(rb.ravel(), ac.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
        r1a, r1b = int(0.7 * s_), min(int(1.4 * s_) + 1, len(rad) - 1); r1 = r1a + int(np.argmax(rad[r1a:r1b + 1])) if r1b > r1a else r1a; r2a, r2b = min(int(1.6 * s_), len(rad) - 2), min(int(2.5 * s_) + 1, len(rad) - 1); h2 = float(rad[r2a:r2b + 1].max()) if r2b > r2a else 0.0
        sh = (RR > r1 - 2) & (RR < r1 + 2); v = np.maximum(ac[sh], 0); ap = np.abs((v * np.exp(6j * AT[sh])).sum()) / (v.sum() + 1e-9)
        out[t] = (f0, h, wd, hs[0], hs[1], hs[2], r1, rad[r1], h2, h2 / (abs(rad[r1]) + 1e-3), ap, S[t].std())
    keep = out[:, 1] >= np.quantile(out[:, 1], 2 / 3); res = []
    for j in range(len(STAT)):
        v = out[:, j]; res += [np.median(v), np.percentile(v, 25), np.percentile(v, 75), np.median(v[keep])]
    res += [len(P) / (((H - T) // ST + 1) * ((W - T) // ST + 1)), float((out[:, 1] > 3).mean())]
    return i, np.array(res, np.float32)


def main():
    workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1))); F = np.full((8954, len(NAMES)), np.nan, np.float32)
    with get_context('fork').Pool(workers) as pool:
        for k, (i, v) in enumerate(pool.imap_unordered(one, range(8954), chunksize=8)):
            F[i] = v
            if (k + 1) % 2000 == 0: print(k + 1, flush=True)
    np.savez(OUT, F=F, names=np.array(NAMES)); print('done', workers, flush=True)


if __name__ == '__main__':
    main()
