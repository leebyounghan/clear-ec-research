# CD density channel: verdict (2026-11-06)

**Killed.** No non-CNN density estimate adds to hybrid B on CD. The pre-registered primary model made CD worse, and the simplest post-hoc blend gains 0.011 at most (0.004 on the mean, below the +-0.02 noise floor).

CPU only, 8954 rows, project folds. Files on the pod: `experiments/cd_density_channel_20261106/` (`protocol.json`, `features.py`, `features.npz`, `evaluate.py`, `evaluate.log`, `result.json`, `posthoc_blend.py`, `posthoc_blend.log`, `posthoc_blend.json`). Scripts also in `clear_ec_ideation_scripts/cd_density/` next to this file.

## Pre-registered gate

Reference: hybrid B + nested stretch, CD 6.264. Kill above CD -0.05 or if the CI includes 0.

| Model | Features | CD change | CI on CD | Folds negative |
|---|---|---:|---|---:|
| Huber, per source (primary) | All | +0.087 | +0.052, +0.122 | 0 of 5 |
| Huber, per source | Fourier | +0.042 | +0.021, +0.063 | 0 of 5 |
| Huber, per source | Peaks | +0.057 | +0.024, +0.090 | 1 of 5 |
| L1 boosting | All | +0.027 | +0.009, +0.047 | 0 of 5 |
| L1 boosting | Fourier | +0.037 | +0.021, +0.053 | 0 of 5 |
| L1 boosting | Peaks | +0.015 | -0.005, +0.036 | 1 of 5 |

Grouping source A's consecutive serials in the bootstrap changes the CI by at most 0.002.

## Post-hoc check, outside the gate

One weight per source on one calibrated estimate, nested, because 75 collinear features could have hidden a real signal.

| Estimate | CD change | CI on CD | Weight A | Weight B |
|---|---:|---|---:|---|
| Detector s48, top 3 clusters, triangle median | -0.011 | -0.018, -0.003 | 0 | 0.06-0.12 |
| Guided Fourier + detector consensus | -0.009 | -0.014, -0.004 | 0 | 0.02-0.04 |
| Detector consensus, top 3 clusters | -0.008 | -0.013, -0.003 | 0 | 0.02-0.06 |
| Guided Fourier ring | 0.000 | 0.000, 0.000 | 0 | 0 |

The CI excludes 0 and all five folds are negative for the detector estimates, but the size is too small to matter.

## Single estimators

Best estimators by CD MAPE (hybrid B + stretch: A 7.10, B 5.42):

| Estimate | A MAPE | B MAPE | Spearman with log CD, A / B | Error Spearman with hybrid B, A / B |
|---|---:|---:|---|---|
| s48, top 3 clusters, triangle median | 8.08 | 6.01 | 0.639 / 0.847 | 0.790 / 0.768 |
| s48, top 1 cluster, triangle median | 8.01 | 6.12 | 0.643 / 0.842 | 0.790 / 0.742 |

Reading: a detector with no challenge labels comes within 0.6 of hybrid B on source B, and its errors follow hybrid B's closely. Two different readers err together. That error could come from the label or from reading the wrong region. This test cannot separate them.

## CD error tail (reference)

| Largest errors | Share of CD error | Points of CD score | Median label | Median prediction | Over-predicted |
|---:|---:|---:|---:|---:|---:|
| 8 frames | 3.8 % | 0.237 | 736 | 2696 | 100 % |
| 35 | 6.9 % | 0.435 | 1176 | 2313 | 94 % |
| 90 (1 %) | 10.5 % | 0.655 | 1783 | 2344 | 80 % |
| 448 (5 %) | 24.4 % | 1.531 | 2336 | 2612 | 58 % |

- CD without the top 8 is 6.033; without the top 35, 5.852.
- Labels below 1500: 138 frames, mean APE 34.7 %, 8.5 % of CD error.
- A top-8 frame averages 265 % APE. In 1000 test frames, about 0.9 such frames are expected, and one contributes about 0.27 to CD. The 0.113 CD gap to first place is smaller than that. Whether first place leads for this reason is unknown.
- The primary model's damage is almost all outside the top 5 %: +0.085 of the +0.087.

## Ledger check

Not a repeat of `freq_lattice_20261105` (128 px tiles, coarse bins) or `p0_cd_estimators_20261028` (other detectors, no fixed-count windows), so it ran. It reached the same conclusion: S15 CD +0.011, p0 +0.007. The session that started this one had not yet answered about tests after 2026-11-05; with the idea killed, the answer no longer changes anything.

## Measurements worth adding to the brief

Measurements only, for the PI or the session holding the brief to decide:

- The two ID formats overlap in time: `NNNN-YY` 2018-2024, `YYYY-NNNN` 2019-2022; mean CV 0.418 and 0.347.
- CV r within a source 0.54-0.56, against 0.695 over all rows.
- Fellow-eye residual Spearman in source B: CD 0.34, CV 0.18, HEX 0.10.
- Source A consecutive serials: label Spearman CD about 0.5; folds split them (same-fold share 0.187).
- CD tail table above.
