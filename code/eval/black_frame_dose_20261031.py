"""Is recipe A's weakness a dose-response in the black-frame fraction (normalisation hypothesis)? CPU, OOF only."""
import json, sys
import numpy as np
import cv2
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

b = E.load_base(); y = b['y']; st = E.image_stats(b)
bf = st['black_frac']; year = np.array([E.year_of(i) for i in b['ids']])
eA, eP = E.ape(b['A'], y), E.ape(b['P1'], y)
d = np.nanmean(eA, 1) - np.nanmean(eP, 1)          # per-image A minus P1 (mean over scored metrics)
bins = [0, 1e-4, 0.01, 0.05, 0.10, 0.20, 1.01]
out = dict(by_black_frac=[], within_years={})
for lo, hi in zip(bins[:-1], bins[1:]):
    r = (bf >= lo) & (bf < hi)
    out['by_black_frac'].append(dict(bin=f'[{lo},{hi})', n=int(r.sum()), A_minus_P1=round(float(d[r].mean()), 3)))
for name, yr in (('2019_2022', (year >= 2019) & (year <= 2022)), ('2018_2023_2024', (year == 2018) | (year >= 2023))):
    out['within_years'][name] = {lab: dict(n=int((yr & m).sum()), A_minus_P1=round(float(d[yr & m].mean()), 3))
                                 for lab, m in (('no_frame', bf <= 0.01), ('frame', bf > 0.01))}
# what the frozen preprocessing does to the cell region: 1/99 percentile window with vs without black pixels
mm = np.memmap('/workspace/clear-ec-research/datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
rng = np.random.default_rng(0)
pick = rng.choice(np.where(bf > 0.10)[0], 40, replace=False)
ratio = []
for n in pick:
    a = cv2.createCLAHE(2.0, (1296 // 32, 972 // 32)).apply(np.ascontiguousarray(mm[b['idx'][n]]))
    lo, hi = np.percentile(a, (1, 99)); fg = a[mm[b['idx'][n]] >= 5]
    lo2, hi2 = np.percentile(fg, (1, 99))
    ratio.append((hi - lo) / (hi2 - lo2 + 1e-6))
out['frame_images_window_ratio_all_vs_foreground'] = dict(median=round(float(np.median(ratio)), 3), q10=round(float(np.quantile(ratio, 0.1)), 3), q90=round(float(np.quantile(ratio, 0.9)), 3))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(out, indent=1))
