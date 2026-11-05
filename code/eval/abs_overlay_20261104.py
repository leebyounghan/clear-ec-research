"""Overlay of the image-fitted boundaries on two annotated frames for a visual check (2026-11-04)."""
import json, sys
import cv2
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import abs_tess_20261104 as T
ann = json.load(open(f'{T.ROOT}/datasets/clear_ec_official/green_eval/annotations.json'))
with np.load(f'{T.ROOT}/datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z: ids = z['ids'].astype(str)
pos = {n: i for i, n in enumerate(ids)}; P = T.load_peaks()
mm = np.memmap(T.RAW, dtype=np.uint8, mode='r', shape=(8954, T.H, T.W))
for n in sys.argv[2:]:
    a = np.asarray(mm[pos[n]]); lab, info = T.tessellate(a, P[pos[n]], 0.35, 0.0, 99.0); e = ann[n]; x0, y0, x1, y1 = e['roi_box_mha']
    bd = np.zeros_like(lab, bool); bd[:, 1:] |= lab[:, 1:] != lab[:, :-1]; bd[1:] |= lab[1:] != lab[:-1]
    rgb = np.dstack([a, a, a]).copy(); rgb[bd] = (0, 255, 255)
    for x, y in e['markers_mha']: cv2.circle(rgb, (int(x), int(y)), 3, (0, 255, 0), -1)
    for x, y, p in P[pos[n]]:
        if p >= 0.35: cv2.drawMarker(rgb, (int(x), int(y)), (255, 80, 80), cv2.MARKER_CROSS, 7, 1)
    c = rgb[y0:y1, x0:x1]; ar = np.dstack([a, a, a])[y0:y1, x0:x1]; out = np.hstack([ar, c])
    cv2.imwrite(f'{sys.argv[1]}/overlay_{n}.png', out[:, :, ::-1]); print(n, info, out.shape)
