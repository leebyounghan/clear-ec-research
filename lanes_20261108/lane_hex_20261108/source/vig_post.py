"""Vigueras-Guillen edge + ROI maps -> cells -> the CPSAM-route features (HEX lane, 2026-11-11).
Cells: a Python re-implementation of the authors' MATLAB postprocessing (obtain_freqAnalysis + applyCellWatershedFreq,
TVST repository): characteristic frequency fc = peak of the radial mean of the edge map's Fourier spectrum (reconstruction
by dilation, two lowest bins and the lowest 4 % removed; no curve fit), 1-px border of 128, Gaussian smoothing sigma 0.2/fc
(zero-padded), 4-connected watershed from the regional minima, then their correction step (seeds = inner 80 % of each cell,
4-connected, seeded watershed on the sigma-2 smoothed map). ROI (TVST note): a cell is kept when >= 75 % of its pixels have
ROI probability >= 0.5; other cells become background (0). The label map is mapped back to the 972 x 1296 frame grid
(nearest) and summarised by feats(), a verbatim copy of seg_run.feats (the CPSAM route), plus fc and the ROI fraction.
usage: vig_post.py MAPS_DIR OUT.npz [workers]      | vig_post.py --check-feats  (copy of feats vs a stored CPSAM chunk)"""
import os, sys, glob, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import norm
from skimage.morphology import reconstruction, local_minima
from skimage.segmentation import watershed, relabel_sequential
from skimage.measure import label

LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
FH, FW = 972, 1296

# ---- verbatim copy of seg_run.py (CPSAM route) feature code ----
R = 3; OFFS = [(dy, dx) for dy in range(-R, R + 1) for dx in range(0, R + 1) if (dx > 0 or dy > 0) and dy * dy + dx * dx <= R * R]
UM2 = (1000 / 1296) ** 2
NAMES = ['h6', 'h5', 'h7', 'n_int', 'area_cv', 'cd_mm2', 'coverage', 'n_cells'] + [f'n{k}' for k in range(3, 10)] + ['h6_strict', 'n_strict']


def feats(M):
    H, W = M.shape; n = int(M.max()); out = [np.nan] * len(NAMES)
    if n < 10: return out
    pairs = []
    for dy, dx in OFFS:
        y0, y1 = max(0, -dy), H - max(0, dy); a = M[y0:y1, 0:W - dx]; b = M[y0 + dy:y1 + dy, dx:W]
        k = (a > 0) & (b > 0) & (a != b); pairs.append(np.c_[a[k], b[k]])
    P = np.unique(np.sort(np.concatenate(pairs), 1), axis=0)
    nb = np.bincount(P.ravel(), minlength=n + 1)
    edge = np.unique(np.r_[M[0], M[-1], M[:, 0], M[:, -1]]); area = np.bincount(M.ravel(), minlength=n + 1).astype(float)
    dil = ndi.grey_dilation(M, size=(2 * R + 1, 2 * R + 1)); ring = (M == 0) & (dil > 0)
    bg = np.bincount(dil[ring], minlength=n + 1).astype(float)
    per = np.bincount(dil[(dil > 0) & (dil != M)], minlength=n + 1).astype(float)
    frac_bg = bg / np.maximum(per, 1)
    ok = np.ones(n + 1, bool); ok[0] = False; ok[edge] = False; ok &= frac_bg < 0.25; ok &= area > 30
    if ok.sum() < 5: return out
    s = nb[ok]; a = area[ok]; med = np.median(a)
    st = ok & (frac_bg < 0.1) & (area > 0.4 * med) & (area < 2.5 * med)
    hs = np.mean(nb[st] == 6) if st.sum() >= 5 else np.nan
    return [np.mean(s == 6), np.mean(s == 5), np.mean(s == 7), float(ok.sum()), a.std() / a.mean(), 1e6 / (a.mean() * UM2),
            float((M > 0).mean()), float(n)] + [float(np.sum(s == k)) for k in range(3, 10)] + [hs, float(st.sum())]
# ---- end of copy ----


NAMES_X = NAMES + ['vig_fc', 'vig_roi_frac']
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def smooth(P, sig):
    h = np.floor(3 * sig); g = norm.pdf(np.arange(-h, h + 1), 0, sig)
    return ndi.convolve1d(ndi.convolve1d(P, g, axis=0, mode='constant'), g, axis=1, mode='constant')


def freq_fc(img):
    N, M = img.shape; Fm = np.abs(np.fft.fftshift(np.fft.fft2(img))) / (N * M)
    gN = np.arange(-N / 2, N / 2) if N % 2 == 0 else np.arange(N) - (N - 1) / 2
    gM = np.arange(-M / 2, M / 2) if M % 2 == 0 else np.arange(M) - (M - 1) / 2
    if N > M: gM = gM * N / M
    elif M > N: gN = gN * M / N
    X, Y = np.meshgrid(gM, gN); rho = np.floor(np.hypot(X, Y) + 0.5).astype(np.int64)
    rec = reconstruction(np.where(rho == 0, Fm, 0.0), Fm, method='dilation', footprint=CROSS); Fr = Fm - rec
    half = max(N, M) // 2 + 1; r = rho.ravel(); keep = r < half
    mode = np.bincount(r[keep], Fr.ravel()[keep], half) / np.maximum(np.bincount(r[keep], minlength=half), 1)
    freq = np.arange(1, half + 1) / half / 2; mode[:2] = 0; fmin = int(np.floor(0.04 * half + 0.5))
    return float(freq[fmin:][np.argmax(mode[fmin:])])


def segment(E8, R8):
    """E8, R8: uint8 edge / ROI maps (prob * 255). Returns the label map on the frame grid (int32) and fc."""
    E = E8.astype(np.float64); N, M = E.shape; fc = freq_fc(E)
    P = np.full((N + 2, M + 2), 128.0); P[1:-1, 1:-1] = E
    G = smooth(P, 0.2 / fc)
    ws = watershed(G, label(local_minima(G, connectivity=1), connectivity=1), connectivity=1, watershed_line=True)[1:-1, 1:-1]
    edges = ws == 0; edges[0, :] = edges[-1, :] = edges[:, 0] = edges[:, -1] = True
    ex = edges.copy(); ex[0, :] = ex[-1, :] = ex[:, 0] = ex[:, -1] = False
    dE = ndi.distance_transform_edt(~ex)
    cl = label(~edges, connectivity=1); n = int(cl.max()); f = cl.ravel(); sz = np.bincount(f, minlength=n + 1)
    rr, cc = np.indices(cl.shape); mr = np.bincount(f, rr.ravel(), n + 1) / np.maximum(sz, 1); mc = np.bincount(f, cc.ravel(), n + 1) / np.maximum(sz, 1)
    dC = np.hypot(rr - mr[cl], cc - mc[cl])
    mark = (cl > 0) & (sz[cl] > 1) & (dC / np.maximum(dC + dE, 1e-12) < 0.8)
    mark &= ndi.convolve(mark.astype(np.uint8), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8), mode='constant') > 0
    Gs = smooth(P, 2.0)[1:-1, 1:-1]
    lab = watershed(Gs, label(mark, connectivity=1), connectivity=1)   # full tessellation; cell boundaries are label changes
    n = int(lab.max()); f = lab.ravel()
    inroi = np.bincount(f, (R8 >= 128).ravel().astype(float), n + 1) / np.maximum(np.bincount(f, minlength=n + 1), 1)
    keep = inroi >= 0.75; keep[0] = False
    lab = relabel_sequential(np.where(keep[lab], lab, 0))[0]
    ri = np.minimum(((np.arange(FH) + 0.5) * N / FH).astype(int), N - 1); ci = np.minimum(((np.arange(FW) + 0.5) * M / FW).astype(int), M - 1)
    return lab[ri][:, ci].astype(np.int32), fc


def seg_and_feats(er):
    """(edge uint8, roi uint8) -> NAMES_X row"""
    Mf, fc = segment(er[0], er[1]); return list(feats(Mf)) + [fc, float((er[1] >= 128).mean())]


def do_chunk(path):
    z = np.load(path); out = np.full((len(z['frames']), len(NAMES_X)), np.nan, np.float32)
    for k in range(len(z['frames'])): out[k] = seg_and_feats((z['edge'][k], z['roi'][k]))
    return z['frames'], out


if __name__ == '__main__':
    if sys.argv[1] == '--check-feats':
        z = np.load(LANE + '/seg_loose/c000.npz'); d = 0.0
        for k in range(8):
            a = np.array(feats(np.asarray(z['masks'][k], np.int32)), float); b = z['F'][k].astype(float); fin = np.isfinite(a) | np.isfinite(b)
            d = max(d, float(np.nanmax(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0)
        print('feats copy vs stored CPSAM chunk, 8 frames: max |diff|', d, flush=True); sys.exit(0)
    from multiprocessing import Pool
    files = sorted(glob.glob(os.path.join(sys.argv[1], 'c*.npz'))); files = [f for f in files if not f.endswith('.tmp.npz')]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 4; t0 = time.time(); fr, F = [], []
    with Pool(nw) as pool:
        for i, (a, b) in enumerate(pool.imap(do_chunk, files)):
            fr.append(a); F.append(b); print(f'chunk {i + 1}/{len(files)} {(time.time() - t0) / sum(len(x) for x in fr):.2f} s/frame wall', flush=True)
    np.savez_compressed(sys.argv[2], frames=np.concatenate(fr), F=np.concatenate(F), names=np.array(NAMES_X)); print('done', sys.argv[2], flush=True)
