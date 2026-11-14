"""Zero-shot Cellpose-SAM segmentation -> HEX from real cell borders (HEX lane, 2026-11-08). No training, local weights only
(cpsam_models/cpsam_v2). Per frame: masks; adjacency = label pairs within R px; interior cells = not touching the frame
edge and with < 25 % background in a ring around them; features: share of interior cells with 6 neighbours (and 5, 7),
interior count, area CV, cells per mm2, mean cellpose probability proxy (mask coverage).
usage: seg_hex.py OUT.npz FRAMES(comma list | 'sample:N:SEED' rows | 'all') [batch]   (frames are frame indices)."""
import sys, time, os
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
os.chdir('/workspace/clear-ec-research')
from cellpose import models
from scipy import ndimage as ndi
OUT, SPEC = os.path.join(LANE, sys.argv[1]), sys.argv[2]; BS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MM = np.memmap('/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
if SPEC == 'all': FR = np.arange(8954)
elif SPEC.startswith('sample:'):
    import hexlib as L
    _, n, sd = SPEC.split(':'); rows = np.random.default_rng(int(sd)).choice(L.N, int(n), replace=False); FR = L.idx[rows]
else: FR = np.array([int(q) for q in SPEC.split(',')])
R = 3; OFFS = [(dy, dx) for dy in range(-R, R + 1) for dx in range(0, R + 1) if (dx > 0 or dy > 0) and dy * dy + dx * dx <= R * R]
UM2 = (1000 / 1296) ** 2
# settings via env: CP_FLOW, CP_PROB (cellpose thresholds), CP_CLAHE=1 (CLAHE clip 3, 8x8 tiles before cellpose)
KW = {k: float(os.environ[e]) for k, e in (('flow_threshold', 'CP_FLOW'), ('cellprob_threshold', 'CP_PROB')) if e in os.environ}
CLAHE = None
if os.environ.get('CP_CLAHE') == '1':
    import cv2; CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def feats(M):
    H, W = M.shape; n = int(M.max())
    if n < 10: return [np.nan] * 8
    pairs = []
    for dy, dx in OFFS:
        y0, y1 = max(0, -dy), H - max(0, dy); a = M[y0:y1, 0:W - dx]; b = M[y0 + dy:y1 + dy, dx:W]
        k = (a > 0) & (b > 0) & (a != b); pairs.append(np.c_[a[k], b[k]])
    P = np.concatenate(pairs); P = np.sort(P, 1); P = np.unique(P, axis=0)
    nb = np.bincount(P.ravel(), minlength=n + 1)
    edge = np.unique(np.r_[M[0], M[-1], M[:, 0], M[:, -1]]); area = np.bincount(M.ravel(), minlength=n + 1).astype(float)
    dil = ndi.grey_dilation(M, size=(2 * R + 1, 2 * R + 1)); ring = (M == 0) & (dil > 0)
    bg = np.bincount(dil[ring], minlength=n + 1).astype(float)
    per = np.bincount(dil[(dil > 0) & (dil != M)], minlength=n + 1).astype(float)
    frac_bg = bg / np.maximum(per, 1)
    ok = np.ones(n + 1, bool); ok[0] = False; ok[edge] = False; ok &= frac_bg < 0.25; ok &= area > 30
    if ok.sum() < 5: return [np.nan] * 8
    s = nb[ok]; a = area[ok]
    return [np.mean(s == 6), np.mean(s == 5), np.mean(s == 7), float(ok.sum()), a.std() / a.mean(), 1e6 / (a.mean() * UM2), float((M > 0).mean()), float(n)]


def main():
    model = models.CellposeModel(gpu=True, pretrained_model='cpsam_models/cpsam_v2')
    F = np.full((len(FR), 8), np.nan, np.float32); t0 = time.time()
    for s in range(0, len(FR), BS):
        fr = FR[s:s + BS]; imgs = [np.asarray(MM[i]) for i in fr]
        if CLAHE is not None: imgs = [CLAHE.apply(im) for im in imgs]
        masks = model.eval(imgs, batch_size=8, **KW)[0]
        for k, M in enumerate(masks): F[s + k] = feats(np.asarray(M, np.int32))
        if s % (BS * 10) == 0: print(s, f'{(time.time() - t0) / (s + len(fr)):.2f} s/frame', np.round(F[s], 3).tolist(), flush=True)
    np.savez_compressed(OUT, frames=FR, F=F, names=np.array(['h6', 'h5', 'h7', 'n_int', 'area_cv', 'cd_mm2', 'coverage', 'n_cells']))
    print('done', len(FR), f'{(time.time() - t0) / len(FR):.2f} s/frame', flush=True)


if __name__ == '__main__':
    main()
