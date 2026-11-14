"""Full-frame zero-shot Cellpose-SAM run, chunked and resumable (HEX lane, 2026-11-08).
Frames in a fixed random order (seed 0), 64 per chunk -> OUTDIR/cNNN.npz with frames, masks (uint16), F (17 features).
GPU duty from LANE/duty_seg.txt (sleep after each batch so GPU time / wall time ~ duty). Settings via env as seg_hex.py
(CP_FLOW, CP_PROB, CP_CLAHE=1). usage: seg_run.py OUTDIR(rel to LANE) [chunk_lo chunk_hi]"""
import sys, time, os
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
os.chdir('/workspace/clear-ec-research')
from cellpose import models
from scipy import ndimage as ndi
OUTD = os.path.join(LANE, sys.argv[1]); os.makedirs(OUTD, exist_ok=True)
CH = 64; ORDER = np.random.default_rng(0).permutation(8954); NCH = (len(ORDER) + CH - 1) // CH
LO = int(sys.argv[2]) if len(sys.argv) > 2 else 0; HI = int(sys.argv[3]) if len(sys.argv) > 3 else NCH
MM = np.memmap('/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
KW = {k: float(os.environ[e]) for k, e in (('flow_threshold', 'CP_FLOW'), ('cellprob_threshold', 'CP_PROB')) if e in os.environ}
CLAHE = None
if os.environ.get('CP_CLAHE') == '1':
    import cv2; CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
DUTYF = os.path.join(LANE, 'duty_seg.txt')
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


def duty():
    try: return min(1.0, max(0.05, float(open(DUTYF).read().strip())))
    except Exception: return 1.0


def main():
    model = models.CellposeModel(gpu=True, pretrained_model='cpsam_models/cpsam_v2'); t0 = time.time(); nd = 0
    for c in range(LO, HI):
        path = os.path.join(OUTD, f'c{c:03d}.npz')
        if os.path.exists(path): continue
        fr = ORDER[c * CH:(c + 1) * CH]; Ms = np.zeros((len(fr), 972, 1296), np.uint16); F = np.full((len(fr), len(NAMES)), np.nan, np.float32)
        for s in range(0, len(fr), 4):
            imgs = [np.asarray(MM[i]) for i in fr[s:s + 4]]
            if CLAHE is not None: imgs = [CLAHE.apply(im) for im in imgs]
            tg = time.time(); masks = model.eval(imgs, batch_size=8, **KW)[0]; tg = time.time() - tg; tc = time.time()
            for k, M in enumerate(masks):
                M = np.asarray(M, np.int32); Ms[s + k] = M; F[s + k] = feats(M)
            tc = time.time() - tc; d = duty()
            if d < 1.0: time.sleep(max(0.0, tg * (1.0 / d - 1.0) - tc))
        tmp = path + '.tmp.npz'; np.savez_compressed(tmp, frames=fr, masks=Ms, F=F, names=np.array(NAMES)); os.replace(tmp, path)
        nd += len(fr); print(f'chunk {c} done, {nd} frames, {(time.time() - t0) / nd:.2f} s/frame wall, last gpu {tg / len(masks):.2f} s/frame, duty {duty()}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
