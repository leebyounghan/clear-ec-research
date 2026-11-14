"""Verbatim copy of lane_hex_20261108/source/seg_run.py feats() and its constants (lines 20-46 of that file, 2026-11-10); seg_run.py itself starts a run at import."""
import numpy as np
from scipy import ndimage as ndi
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


