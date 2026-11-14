"""Pixel-scale check (backbone lane, 2026-11-11): dominant cell-pattern frequency of 256 px crops (CLAHE, high-pass, radially averaged
power spectrum peak in 1/60..1/8 cycles/px), calibrated on synthetic frames with known spacing d (fit d = c / f*), applied to real
frames with their CD label to get the implied um/px: A_px = (sqrt3/2) d^2, um/px = sqrt(1e6 / (CD * A_px)). Measurement only."""
import sys, os, json
os.chdir('/workspace/clear-ec-research'); sys.path[:0] = ['experiments/oof_recipeA_20261027/staging_regression_zero', 'experiments/phase2_crossfold_20261015/source', 'experiments/lane_backbone_20261108/source']
import numpy as np, cv2
import phase2_foundation_train_20261002 as b
import synth_specular_20261111 as G
def peak_freq(a):
    c = cv2.createCLAHE(2.0, (8, 8)).apply(a).astype(np.float32); c = c - cv2.GaussianBlur(c, (0, 0), 10)
    c = c * np.outer(np.hanning(c.shape[0]), np.hanning(c.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2; h, w = P.shape; yy, xx = np.indices(P.shape); r = np.hypot(yy - h // 2, xx - w // 2)
    rb = np.arange(4, 40); prof = np.array([P[(r >= q - 0.5) & (r < q + 0.5)].mean() for q in rb])
    # whiten: remove the 1/f^a trend (log-log linear fit), take the strongest bump above it
    co = np.polyfit(np.log(rb), np.log(prof), 1); res = np.log(prof) - np.polyval(co, np.log(rb))
    k = int(np.argmax(res)); return rb[k] / w  # cycles per px
crops = [(358, 520), (200, 300), (500, 800), (300, 900)]
# calibration on synthetic frames
cal = []
for i in range(40):
    img, lab, prm = G.render(1000 + i)
    f = np.median([peak_freq(np.ascontiguousarray(img[y:y + 256, x:x + 256])) for y, x in crops]); cal.append((prm['spacing_px'], f))
cal = np.array(cal); c = float(np.median(cal[:, 0] * cal[:, 1]))
with np.load(b.META, allow_pickle=True) as z:
    n = len(z['ids']); Y = z['Y'].astype(float)
raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(n, 972, 1296))
sp = np.load('experiments/lane_backbone_20261108/data/source_split_20261110.npz', allow_pickle=True)
rng = np.random.default_rng(7); res = {}
for nm, P in (('P1', sp['P1']), ('P2', sp['P2'])):
    um = []; dd = []
    for i in rng.choice(P, 150, replace=False):
        i = int(i)
        if not (Y[i] > 0).all():
            continue
        f = np.median([peak_freq(np.ascontiguousarray(raw[i][y:y + 256, x:x + 256])) for y, x in crops]); d = c / f; dd.append(d)
        um.append(np.sqrt(1e6 / (Y[i, 0] * (np.sqrt(3) / 2) * d * d)))
    res[nm] = dict(n=len(um), spacing_px_p25_50_75=np.percentile(dd, [25, 50, 75]).round(2).tolist(), um_per_px_p25_50_75=np.percentile(um, [25, 50, 75]).round(3).tolist())
print(json.dumps(dict(calibration_c=round(c, 3), calib_spread=np.percentile(cal[:, 0] * cal[:, 1], [10, 90]).round(3).tolist(), assumed_um_per_px=round(G.UM, 4), **res)))
