# E4 — error decomposition on the 25 expert marker sets (2026-11-03, 16:06 UTC; sources section appended when E1 lands)

Pre-registered 16:10 UTC (protocol.json E4). Oracle definition = p1 (bounded Voronoi cells inside the marker hull,
CD = 1e6 / mean area, CV = std/mean area, HEX = six-sided fraction), LOO median-ratio scaling, 8 levels x 30 repeats
per axis, 1380 tasks, 140 s on a 2-worker pool (load was 15.5). B on the same 25 images: CD 7.66 / CV 17.93 / HEX 11.45.
Oracle at zero degradation: CD 7.23 / CV 8.84 / HEX 7.75 (r 0.75 / 0.88 / 0.84), 100 bounded cells per image.

## LOO-scaled MAPE on the 25 (oracle definition; complete-cell filter in brackets for CV / HEX)

| axis / level | CD | CV | HEX |
|---|---|---|---|
| misses 5 / 10 / 20 / 30 / 40 % | 7.8 / 7.9 / 8.3 / 9.2 / 9.7 | 11.5 / 13.5 / 16.5 / 16.4 / 18.3 [12.9 / 14.0 / 16.5 / 18.3 / 19.4] | 9.9 / 12.3 / 14.7 / 18.0 / 20.3 [9.9 / 12.5 / 16.3 / 20.1 / 25.0] |
| false positives 5 / 10 / 20 / 30 / 40 % (boundary / near-cell) | 8.2 / 9.2 / 10.7 / 11.7 / 12.3 | 36.8 / 64.3 / 88.0 / 102.6 / 98.9 [14.5 / 17.3 / 18.2 / 19.3 / 20.0] | 9.2 / 10.3 / 13.3 / 14.5 / 15.8 [8.5 / 9.9 / 13.3 / 15.3 / 20.3] |
| jitter 0.1 / 0.2 / 0.3 / 0.4 x spacing | 7.4 / 7.4 / 7.7 / 7.7 | 8.8 / 11.4 / 13.9 / 15.7 [11.7 / 15.1 / 18.5 / 21.4] | 9.2 / 12.4 / 17.8 / 18.7 [9.3 / 13.2 / 22.7 / 39.6] |
| window IoU 0.9 / 0.7 / 0.5 (partial coverage, 40 / 32 / 22 cells) | 8.4 / 8.4 / 8.7 | 15.8 / 15.7 / 19.5 | 13.7 / 14.9 / 17.3 |

Miss x FP grid (oracle CV / HEX): FP 0.1 alone 66.9 / 10.4; miss 0.1 alone 13.2 / 12.3; (0.1, 0.1) 51.9 / 13.0;
(0.2, 0.2) 50.8 / 16.9; (0.3, 0.3) 63.1 / 18.4.

## Break-even per axis (level at which the oracle estimator stops beating B on the 25)

| metric | misses | false positives | jitter (x spacing) | window IoU |
|---|---|---|---|---|
| CD (B 7.66) | ~4 % | ~2 % | ~0.29 (marginal: 7.60 at 0.25, 7.68 at 0.3) | already above at 0.9 (sample size) |
| CV (B 17.93) | ~36 % | ~2 % (oracle def.) / ~11 % (complete-cell filter) | never (15.7 at 0.4) | ~0.55 |
| HEX (B 11.45) | ~8 % | ~14 % | ~0.17 | already above at 0.9 (sample size) |

## Reading (simulation part)

* **CV is dominated by false positives of the split / near-cell kind**: 5 % of them quadruple the CV error (36.8 vs 8.8)
  because each creates a sliver cell; random misses barely hurt CV (36 % before break-even) and jitter never breaks it.
  So the lever for CV is a *split-aware / robust area estimator* (reject sliver cells, merge near-duplicates, mixture
  deconvolution), not recall. The simple complete-cell filter already moves the FP break-even from 2 % to 11 %.
* **HEX is dominated by misses, then jitter**: 8 % random misses or 0.17 x spacing (about 4 px) of localisation noise
  already lose the oracle's advantage over B; FPs are tolerated to ~14 %. HEX therefore needs recall >= 92 % with
  sub-4-px localisation inside the measured region, which is detection-accuracy work.
* **CD has no headroom**: the oracle itself (7.23) is within 0.4 pp of B (7.66) and any error pushes it over.
* The window axis mostly measures sample size (the annotation covers only the ROI, so a shifted window keeps 40
  cells or fewer); it says that fewer than ~60 complete cells per image is itself enough to lose against B.

(Real point sources on the same axes: see the sources section below once E1's CPSAM centroids exist.)

## Real point sources on the same axes (16:35 UTC; region = convex hull of the expert markers dilated by 0.5 spacing, match radius 0.5 x median NN spacing)

| source | subset | miss rate | FP rate | jitter (x spacing) | window IoU vs expert ROI |
|---|---|---|---|---|---|
| fine-tuned stride-2 s46 / s47 / s48 / s49 / s50 | val-5 (clean) | 0.29 / 0.22 / 0.26 / 0.19 / 0.20 | 0.19 / 0.22 / 0.24 / 0.25 / 0.28 | 0.20 / 0.21 / 0.24 / 0.21 / 0.25 | 0.71 |
| same | all-25 (20 leaked) | 0.11 / 0.11 / 0.13 / 0.10 / 0.12 | 0.20 / 0.23 / 0.22 / 0.21 / 0.25 | 0.20-0.24 | 0.79 |
| pre-only s46 / TTA s46 / TTA s49 | val-5 | 0.17 / 0.18 / 0.21 | 0.17 / 0.17 / 0.16 | 0.22 / 0.23 / 0.23 | 0.71 |
| CPSAM v2 centroids | val-5 (all-25) | 0.51 (0.39) | 0.07 (0.06) | 0.23 (0.22) | 0.71 |

Break-even recap (oracle definition): HEX misses ~8 %, jitter ~0.17; CV FP ~2 % (plain) / ~11 % (complete-cell filter).

## Verdict

* **HEX is detection-accuracy-bound and every real source is far from it.** The best clean miss rate is 17-19 % (pre-only,
  s49) against an 8 % break-even, and localisation noise is 0.20-0.25 x spacing (5-6 px) against 0.17 (4 px). Jitter
  alone already costs the oracle's advantage; misses double it. CPSAM is worse on misses (51 %) despite few FPs.
  Reaching recall >= 92 % at < 4 px inside the measured region is a detector-training problem (more in-domain labels,
  localisation-aware loss), not an estimator problem.
* **CV is false-positive-bound and the robust estimator is the right lever (E5).** Real FP rates are 16-28 %, ten
  times the plain-estimator break-even (2 %). E5's tuned estimator (merge < 0.6 spacing, sliver < 0.5 x local median,
  10 % trimmed CV) keeps the 25-image LOO MAPE below B for FP up to 40 % and misses up to 32 % on single axes; on the
  miss x FP grid it is 16.4 at (0.2, 0.2) and 18.8 at (0.3, 0.3) against B's 17.9, i.e. the real sources sit at the
  edge of the beatable region on the 25 images. Whether that survives on the 8954 is E5 step 2 (`e5_robust_cv/apply.json`).
* The window axis says a measured region with fewer than ~60 complete cells loses to B on its own; the detector
  cluster window overlaps the expert ROI at IoU 0.71 on clean images.

Files: `e4_degradation/{simulate.json, sources.json, verdict.md}`, `source/green_day3_e4_degradation_20261103.py`.
