# E5 — split-aware robust area estimator for CV (2026-11-03, 16:22 tune / 16:55 UTC apply): FAIL on real points (w = 0 everywhere)

## Step 1 — tuning on the E4 degradation grid only (10 repeats, 160 variants, 497 s on 2 workers)

Selection = mean LOO-scaled CV MAPE over the FP axis (0.05-0.4) + the 16 miss x FP cells + zero degradation, subject
to zero-degradation MAPE <= 10. Chosen: merge points closer than 0.6 x NN spacing, drop cells with area < 0.5 x the
median of their Delaunay neighbours, no complete-cell filter, **10 % trimmed CV**. Score 14.5 vs plain 57.7; zero
degradation 9.19 vs 8.84. The trimmed statistic does almost all the work (plain trim10 scores 14.9).

| axis | plain CV, break-even vs B (17.93) | chosen, break-even |
|---|---|---|
| false positives | ~1.5 % | never (16.3 at 40 %) |
| misses | ~39 % | ~32 % |
| jitter (x spacing) | never | ~0.39 |
| miss x FP grid (0.1,0.1) / (0.2,0.2) / (0.3,0.3) | 51.9 / 50.8 / 63.1 | 13.8 / 16.4 / 18.8 |

On the 25 images the real detectors' error levels (miss 0.17-0.29, FP 0.16-0.28, jitter 0.2) fall at the edge of the
region where the chosen estimator still beats B.

## Step 2 — frozen estimator on the real fold-clean points (8954 rows, 116,402 estimator calls, 831 s on 4 workers)

| source (window / all cells) | est. CV MAPE | r(err, B err) | LOFO w | delta |
|---|---|---|---|---|
| stride-8 s46 / s47 / s48 / multi | 13.7-14.1 | 0.78-0.79 | 0 | 0.0000 |
| stride-2 s46 / s47 / s48 / s49 / s50 | 13.1-13.5 | 0.80-0.83 | 0 | 0.0000 |
| pre-only s46, TTA s46, TTA s49 | 12.9-13.2 | 0.81-0.82 | 0 | 0.0000 |
| CPSAM centroids / CPSAM mask areas | 12.7-13.1 / 13.0-13.1 | 0.81-0.83 | 0 | 0.0000 |
| mean over the ten / over the full-frame sources | 13.1-13.2 / 12.8-13.1 | 0.81-0.83 | 0 | 0.0000 |

The robust estimator lowers the standalone CV error from ~14.3 (p9 consensus Voronoi CV) to 12.7-13.2 on the
population, but the error that remains is 0.8-correlated with B's own CV error and larger than it (B 10.2), so no fold
gives it weight. Reading: on the 8954 the dominant CV error is not the split-type false positive the simulation
modelled (which trimming removes) but image-level difficulty shared with the regressor (focus, contrast, framing);
the 25 annotated images are too clean to show it. Gates fail (evidence and adoption). CV from points is closed:
oracle 8.8 needs near-perfect points, robustification cannot bridge the real error structure, and E6's learned areas
(w = 0) do not either.

Files: `e5_robust_cv/{tune.json, apply.json, features.npz, tune.log, apply.log}`, `source/green_day3_e5_robust_cv_20261103.py`.
