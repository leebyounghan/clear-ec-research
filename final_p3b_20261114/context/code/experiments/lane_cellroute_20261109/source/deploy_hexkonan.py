"""Deploy path of the cell-route HEX step (kept: HEX 9.7315 vs 9.7380 five-fold OOF on the HEX lane chain_cvw3).
Per image, inputs (all produced upstream, no new model here):
  pk46, pk47, pk48 : (n,3) x, y, conf peaks of the three pre-only sg2 detectors (seeds 46/47/48), each the mean of 4 exact flips (TTA),
                     peaks as SG.peaks2 (conf >= 0.2, NMS 12 px) in raw 1296x972 frame pixels;
  pk3              : the same peak finder on the mean heatmap of the three detectors x 4 flips (the 'seeds3_tta4' peaks);
  seg              : (m,3) centroid x, y and area (px) of the zero-shot Cellpose-SAM masks (cpsam_v2, flow 0.8, cellprob -2, CLAHE clip 3 8x8),
                     the HEX lane's own segmentation run.
features(): frame spacing s = median NN distance of pk3 >= 0.5 (>= 0.35 if <= 20 such peaks); union point set = cross-source vote (0.45 s) of
  pk46/47/48 >= 0.35 and seg centroids with area >= 0.15 s^2, >= 1 source; Konan cells (k=0 Voronoi, k=2 weighted bisectors) over the whole
  frame, interior seeds; statistics over the 3 legible 380x300 windows (most pk3 >= 0.5 peaks) and over the full frame:
  [cv_k0, cv_k2, hex_k0, hex_k2, n] x (win, full).
correct(): ridge on the log HEX residual of the base HEX (the HEX lane's final HEX), parameters from fit_deploy_hexkonan.py (all folds).
Rows with < 50 full-frame cells or non-finite features keep the base HEX."""
import json
import numpy as np
from scipy.spatial import cKDTree
import cr_lib as L
H, W = 972, 1296
NAMES = [f'P2_{r}_{c}' for r in ('win', 'full') for c in ('cv_k0', 'cv_k2', 'hex_k0', 'hex_k2', 'n')]


def windows(hi, w, h, k=3):
    """Top windows by peak count, stride 16, greedy non-overlap (copy of legible_window_features_20261105.windows)."""
    G = np.zeros((H // 16 + 2, W // 16 + 2)); ix = np.minimum((hi[:, 1] / 16).astype(int), G.shape[0] - 1); iy = np.minimum((hi[:, 0] / 16).astype(int), G.shape[1] - 1); np.add.at(G, (ix, iy), 1)
    I = np.zeros((G.shape[0] + 1, G.shape[1] + 1)); I[1:, 1:] = G.cumsum(0).cumsum(1)
    nh, nw = h // 16, w // 16; S = I[nh:, nw:] - I[:-nh, nw:] - I[nh:, :-nw] + I[:-nh, :-nw]; ymax = (H - h) // 16; xmax = (W - w) // 16; S = S[:ymax + 1, :xmax + 1]; out = []; S = S.copy()
    for _ in range(k):
        q = int(np.argmax(S)); y0, x0 = np.unravel_index(q, S.shape); out.append((int(x0) * 16, int(y0) * 16)); a0, a1 = max(0, y0 - nh // 2), min(S.shape[0], y0 + nh // 2 + 1); b0, b1 = max(0, x0 - nw // 2), min(S.shape[1], x0 + nw // 2 + 1); S[a0:a1, b0:b1] = -1
    return out


def features(pk3, pk46, pk47, pk48, seg):
    out = [np.nan] * len(NAMES)
    if pk3 is None or len(pk3) < 20: return np.array(out)
    pk3 = np.asarray(pk3, float); hi = pk3[pk3[:, 2] >= 0.5, :2]; mid = pk3[pk3[:, 2] >= 0.35, :2]; ref = hi if len(hi) > 20 else mid
    if len(ref) < 10: return np.array(out)
    s = float(np.median(cKDTree(ref).query(ref, k=2)[0][:, 1])); wins = windows(ref, 380, 300, k=3)
    srcs = [np.asarray(x, float) for x in (pk46, pk47, pk48)]; srcs = [x[x[:, 2] >= 0.35] for x in srcs]
    sg = np.asarray(seg, float).reshape(-1, 3); sg = sg[sg[:, 2] >= 0.15 * s ** 2]; srcs.append(np.c_[sg[:, :2], np.full(len(sg), 0.5)])
    p, w = L.vote(srcs, s, 0.45, 1); ix, res, nb = L.konan_cells_idx(p)
    inw = np.zeros(len(p), bool)
    for x0, y0 in wins: inw |= (p[:, 0] >= x0) & (p[:, 0] < x0 + 380) & (p[:, 1] >= y0) & (p[:, 1] < y0 + 300)
    vals = []
    for m in (inw, np.ones(len(p), bool)):
        sel = m[ix] if len(ix) else np.zeros(0, bool)
        if sel.sum() < 10: vals += [np.nan] * 4 + [float(sel.sum())]; continue
        a0, s0 = res[0][0][sel], res[0][1][sel]; a2, s2 = res[2][0][sel], res[2][1][sel]
        vals += [a0.std() / a0.mean(), a2.std() / a2.mean(), float(np.mean(s0 == 6)), float(np.mean(s2 == 6)), float(sel.sum())]
    return np.array(vals)


def design(F, hex_base):
    """F: (N,10) features in NAMES order; returns the ridge design (N,11) and the usable-row mask."""
    lg = lambda a: np.log(np.maximum(a, 1e-4)); cols = []
    for r in range(2):
        b = 5 * r; cols += [lg(F[:, b]), lg(F[:, b + 1]), F[:, b + 2], F[:, b + 3], np.log1p(np.nan_to_num(F[:, b + 4]))]
    X = np.column_stack(cols + [np.log(hex_base)]); m = np.isfinite(X).all(1) & (np.nan_to_num(F[:, 9]) >= 50)
    return np.where(m[:, None], X, 0.0), m


def correct(hex_base, F, params):
    hex_base = np.asarray(hex_base, float); X, m = design(np.atleast_2d(F), hex_base)
    Z = (X - np.array(params['mu'])) / np.array(params['sd']); q = hex_base * np.exp(Z @ np.array(params['w'])) * params['c'] / params['c0']
    return np.where(m, q, hex_base)


def seg_points(mask):
    """(972, 1296) integer label mask from the HEX lane's Cellpose-SAM run -> (m, 3) centroid x, y and area px (as source/seg_centroids.py)."""
    from scipy import ndimage as ndi
    lab = np.unique(mask); lab = lab[lab > 0]
    if len(lab) == 0: return np.zeros((0, 3))
    one = np.ones_like(mask, np.float32); cy, cx = np.array(ndi.center_of_mass(one, mask, lab)).T; ar = ndi.sum(one, mask, lab)
    return np.c_[cx, cy, ar].astype(np.float32)
