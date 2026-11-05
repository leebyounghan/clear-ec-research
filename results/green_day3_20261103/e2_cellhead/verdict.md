# E2 — learned per-cell geometry from local appearance (2026-11-03, 16:38 UTC): gates FAIL (member gain -0.0048, just short)

CNN on 96x96 patches around points, trained on expert-point Voronoi labels (hull-bounded cells): srep Topcon 307 images
27,677 cells + the 20 train-green images 2,011 cells (x5), 30 epochs, 108 s. Heads: P(six-sided), log area.

## Sanity (pre-registered: side agreement >= 70 % and area r >= 0.5 on the 5 validation-green images)

| split | six-sided accuracy | log-area r |
|---|---|---|
| srep validation (10 % of images held out) | 0.727 | 0.933 |
| CLEAR-EC validation-green (5 images, expert points) | 0.655 (base rates 0.32-0.42) | 0.449 |

The head transfers poorly to CLEAR-EC appearance: in-domain sanity fails on both outputs (only the srep validation
passes). Everything below is therefore flagged.

## Features on fold-clean detector points (LOFO on B)

| feature | points | target | plain (mean pp) | w | est. MAPE | r(err, B) | geometric hex6 on the same points |
|---|---|---|---|---|---|---|---|
| mean P(six) in the cluster window | sg2_s46 (compact) | HEX | -0.0038 (5/5) | 0.08-0.10 | 11.44 | 0.86 | -0.0065 |
| mean P(six), all cells | sg2_s46 | HEX | -0.0044 (5/5) | 0.08-0.10 | 11.39 | 0.86 | |
| mean P(six) in window | sg2pre_s46 (full frame) | HEX | -0.0023 (4/5) | 0.06-0.08 | 11.46 | 0.86 | -0.0089 |
| fraction P(six) > 0.5 | either | HEX | ~0 | 0-0.04 | | | |
| CV of predicted areas (window / all, 5-95 % trimmed) | either | CV | +0.0004 / 0.0000 (w ~0) | 0-0.02 | 13.4-13.7 | 0.80 | |

Member test in the active ten: + learned(sg2_s46) -0.0345, + learned(sg2pre_s46) -0.0348, + both -0.0359 vs control
-0.0311 (gain -0.0048, 5/5 folds, fold 4 -0.0097) — 0.0002 short of the 0.005 evidence gate, far from 0.02. The learned
P(six) carries some information independent of the geometric hex6 (more than TTA or CPSAM members did), but the
in-domain sanity failure says it is not trustworthy as built; a version trained with more CLEAR-EC labels would be the
way to find out. CV from predicted areas carries nothing (E6 tests the same idea at precision-oriented points).

Files: `e2_cellhead/{result.json, train_result.json, features.npz, labels.npz, labels_val5.npz, model.pt, infer/}`, `source/green_day3_e2_cellhead_20261103.py`.
