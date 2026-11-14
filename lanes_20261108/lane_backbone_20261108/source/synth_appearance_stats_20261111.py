"""Appearance statistics, real vs synthetic (backbone lane, 2026-11-11; measurement only). For each frame, of four 256 px crops the one
with the most cell-band energy (the legible crop) is measured: hf_ratio = power at periods 4-8 px / power at periods 16-32 px (edge
sharpness above the cell frequency; radially averaged spectrum of the raw crop minus a sigma-16 blur); cell_c = std of DoG(2, 8) / mean
(cell-scale contrast); supra_c = std of DoG(8, 24) / mean (cell-to-cell brightness steps + local shading). Usage: real N | synth N [first]"""
import sys, os, json
os.chdir('/workspace/clear-ec-research'); sys.path[:0] = ['experiments/oof_recipeA_20261027/staging_regression_zero', 'experiments/phase2_crossfold_20261015/source', 'experiments/lane_backbone_20261108/source']
import numpy as np, cv2
CROPS = [(358, 520), (200, 300), (500, 800), (300, 900), (150, 600), (600, 400)]
yy, xx = np.indices((256, 256)); RR = np.hypot(yy - 128, xx - 128)
def stats(a):
    best = None
    for y, x in CROPS:
        c = np.asarray(a[y:y + 256, x:x + 256], np.float32)
        if c.mean() < 15:
            continue
        hp = c - cv2.GaussianBlur(c, (0, 0), 16); P = np.abs(np.fft.fftshift(np.fft.fft2(hp * np.outer(np.hanning(256), np.hanning(256))))) ** 2
        per = 256 / np.maximum(RR, 1e-6)
        cell = P[(per >= 16) & (per <= 32)].mean(); hf = P[(per >= 4) & (per <= 8)].mean()
        d1 = cv2.GaussianBlur(c, (0, 0), 2) - cv2.GaussianBlur(c, (0, 0), 8); d2 = cv2.GaussianBlur(c, (0, 0), 8) - cv2.GaussianBlur(c, (0, 0), 24)
        s = (cell, hf / cell, d1.std() / c.mean(), d2.std() / c.mean())
        if best is None or s[0] > best[0]:
            best = s
    return best[1:] if best else None
def summarise(rows):
    R = np.array([r for r in rows if r is not None])
    return {k: np.percentile(R[:, j], [10, 25, 50, 75, 90]).round(4).tolist() for j, k in enumerate(('hf_ratio', 'cell_c', 'supra_c'))}
mode, N = sys.argv[1], int(sys.argv[2])
if mode == 'real':
    import phase2_foundation_train_20261002 as b
    with np.load(b.META, allow_pickle=True) as z:
        n = len(z['ids'])
    raw = np.memmap(b.RAW, mode='r', dtype=np.uint8, shape=(n, 972, 1296)); sp = np.load('experiments/lane_backbone_20261108/data/source_split_20261110.npz', allow_pickle=True)
    rng = np.random.default_rng(11); out = {}
    for nm, P in (('P1', sp['P1']), ('P2', sp['P2'])):
        out[nm] = summarise([stats(raw[int(i)]) for i in rng.choice(P, N, replace=False)])
    print(json.dumps(dict(real=out)))
else:
    import synth_specular_20261111 as G
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    print(json.dumps(dict(synth=summarise([stats(G.render(first + i)[0]) for i in range(N)]))))
