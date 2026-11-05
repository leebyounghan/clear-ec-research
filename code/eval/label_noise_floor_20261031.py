"""Bracket the label-noise floor of the CLEAR-EC ground truth (2026-10-31, CPU only).

Lower bound: sampling noise of an ROI measurement. In each of the 25 train_green images, the technician's cell
markers inside the ROI are tessellated (Voronoi); the ROI is split in two halves (left/right and top/bottom) and
the metric is computed on each half. With independent halves, SD(full ROI) = SD(half1 - half2) / 2. This ignores
spatial heterogeneity across the frame, the device's own segmentation noise and operator variability, so the true
floor is at least this large. GT rounding (CV/HEX to 2 decimals) is added in quadrature.
Upper bound: error shared by two very different predictors (whole-frame recipe A vs crop-based Phase-1 five) on
the 8,954 fold-clean OOF rows. Label noise is part of the shared error, so floor <= the shared component.
MAPE-equivalents use each quantity's SD scaled by the hybrid's own empirical MAPE/SD ratio (heavy-tail aware).
Side check: donors (slide_id, two eyes) split across folds.
"""
import csv, json, re, sys
from collections import defaultdict
import numpy as np
from scipy.spatial import Voronoi

ROOT = '/workspace/clear-ec-research/'
ann = json.load(open(ROOT + 'datasets/clear_ec_official/green_eval/annotations.json'))
gt_rows = {r['ID'].strip(): r for r in csv.DictReader(open(ROOT + 'datasets/clear_ec_official/final_train_ids.csv'))}
UM_PER_PX = 1000 / 1296


def cells(a):
    if 'markers_mha' in a:
        pts = np.asarray(a['markers_mha'], float)
        x0, y0, x1, y1 = a['roi_box_mha']
    else:
        g = np.asarray(a['markers_green'], float)
        bg, bm = a['roi_box_green'], a['roi_box_mha']
        sx = (bm[2] - bm[0]) / (bg[2] - bg[0]); sy = (bm[3] - bm[1]) / (bg[3] - bg[1])
        pts = np.c_[(g[:, 0] - bg[0]) * sx + bm[0], (g[:, 1] - bg[1]) * sy + bm[1]]
        x0, y0, x1, y1 = bm
    vor = Voronoi(pts)
    nb = np.zeros(len(pts), int)
    for p, q in vor.ridge_points:
        nb[p] += 1; nb[q] += 1
    out = []
    for i, r in enumerate(vor.point_region):
        reg = vor.regions[r]
        if not reg or -1 in reg:
            continue
        v = vor.vertices[reg]
        if (v[:, 0] < x0).any() or (v[:, 0] > x1).any() or (v[:, 1] < y0).any() or (v[:, 1] > y1).any():
            continue
        x, y = v[:, 0], v[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * UM_PER_PX ** 2
        out.append((pts[i, 0], pts[i, 1], area, nb[i]))
    return np.array(out)


def metrics(c):
    a = c[:, 2]
    return np.array([1e6 / a.mean(), a.std() / a.mean(), np.mean(c[:, 3] == 6)])


per_image, rel_half = [], []
for name, a in ann.items():
    c = cells(a)
    full = metrics(c)
    d = []
    for ax in (0, 1):
        m = np.median(c[:, ax])
        h1, h2 = metrics(c[c[:, ax] < m]), metrics(c[c[:, ax] >= m])
        d.append((h1 - h2) / ((h1 + h2) / 2))
    rel_half.extend(d)
    g = gt_rows.get(name)
    per_image.append(dict(id=name, n_bounded=int(len(c)), full_vor=[round(float(v), 4) for v in full],
                          gt=None if g is None else [float(g['CD']), float(g['CV']), float(g['HEX'])],
                          half_rel_diff=[[round(float(v), 4) for v in x] for x in d]))
rel_half = np.array(rel_half)                      # (2 * 25, 3)
sd_full_rel = np.sqrt(np.mean(rel_half ** 2, 0)) / 2
gt = np.array([p['gt'] for p in per_image if p['gt']])
round_rel = 0.005 / np.sqrt(3) / gt.mean(0) * np.array([0, 1, 1])   # uniform +-0.005 on CV/HEX

z = np.load(ROOT + 'experiments/recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True)
y, A, P, B, fold, ids = z['target'], z['A_recipeA_tta_scalar'], z['historical_five_calibrated'], z['B_hybrid_w0p5'], z['fold'], z['ids']
ub, mape_B, ratio = {}, {}, {}
for j, k in enumerate(['CD', 'CV', 'HEX']):
    s = y[:, j] > 0
    eA, eP, eB = (np.log(M[s, j] / y[s, j]) for M in (A, P, B))
    cov = np.cov(eA, eP)[0, 1]
    mape_B[k] = float(np.mean(np.abs(B[s, j] / y[s, j] - 1)) * 100)
    ratio[k] = mape_B[k] / (eB.std() * 100)                       # empirical MAPE per unit SD
    ub[k] = dict(shared_var_fraction_of_B=float(cov / eB.var()), corr_A_P=float(np.corrcoef(eA, eP)[0, 1]),
                 floor_upper_mape=float(ratio[k] * np.sqrt(cov) * 100))
lb = {k: dict(roi_sampling_sd_rel=float(sd_full_rel[j]), rounding_sd_rel=float(round_rel[j]),
              floor_lower_mape=float(ratio[k] * np.sqrt(sd_full_rel[j] ** 2 + round_rel[j] ** 2) * 100),
              floor_lower_mape_gaussian=float(0.7979 * np.sqrt(sd_full_rel[j] ** 2 + round_rel[j] ** 2) * 100))
      for j, k in enumerate(['CD', 'CV', 'HEX'])}

# hybrid OOF on the 25 marker images (context only; n=25)
pos = {str(i).strip(): n for n, i in enumerate(ids)}
rows25 = [pos[p['id']] for p in per_image if p['id'] in pos]
mape25 = {k: float(np.mean(np.abs(B[rows25, j] / y[rows25, j] - 1)) * 100) for j, k in enumerate(['CD', 'CV', 'HEX'])}

# donors split across folds
def norm(i):
    return re.sub(r'\s+', ' ', str(i).strip())
slide = {norm(k): r['slide_id'].strip() for k, r in gt_rows.items()}
donor_folds = defaultdict(set)
for i, f in zip(ids, fold):
    sid = slide.get(norm(i))
    if sid:
        donor_folds[sid].add(int(f))
two_eye = [v for v in donor_folds.values()]
split = sum(1 for v in two_eye if len(v) > 1)

res = dict(created='2026-10-31', n_images=len(per_image), bounded_cells_median=float(np.median([p['n_bounded'] for p in per_image])),
           lower_bound=lb, upper_bound=ub, hybrid_B_oof_mape=mape_B, hybrid_B_mape_on_25_marker_images=mape25,
           donors=dict(n_donor_groups_mapped=len(donor_folds), donors_spanning_two_folds=split,
                       donors_with_two_rows=sum(1 for v in donor_folds.values() if True)),
           caveats=['lower bound uses Voronoi metrics of the technician markers as a proxy for the device metrics (CV_vor is biased ~0.49x but linear; relative fluctuations assumed comparable)',
                    'lower bound ignores ROI placement heterogeneity, device segmentation noise, operator variability',
                    'upper bound attributes all A/P1 shared error to labels; shared model bias would also sit there'],
           per_image=per_image)
out = sys.argv[1]
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('n_images', 'bounded_cells_median', 'lower_bound', 'upper_bound', 'hybrid_B_oof_mape', 'hybrid_B_mape_on_25_marker_images', 'donors')}, indent=1))
