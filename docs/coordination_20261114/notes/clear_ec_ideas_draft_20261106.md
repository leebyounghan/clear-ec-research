# CLEAR-EC Phase II ideas: draft for the PI (2026-11-06)

Status: the PI chose idea 1 only, and it was killed the same day: see `clear_ec_cd_density_verdict_20261106.md`. Ideas 2-7 are not agreed and not checked against the ledger. Written from `docs/BRIEF.md` and new CPU measurements on the label table and the out-of-fold (OOF) predictions.
Ledger check not done yet: it runs after the PI agrees the list. Records up to 2026-11-05 are in `git show f8bc49b:<path>`, because `archive_20261106` was deleted; later tests go through the session that started this one. Not opened yet: those records, `green_track_gpu3_notes.md`. Seen while locating the aligned loader: file names in `meta_ceiling_20261103/source/`, a few code lines, and the stored predictions in `sweep/combo_preds.npz`. No result reports.

## What the target means

- Projected best 8.83 against 8.7805: gap 0.149 summed over metrics, of which CD 0.113, CV 0.022, HEX 0.014.
- The CV gap is below test noise (about +-0.06 on the mean, paired, 1000 rows). The CD gap is the only one above it on its own.
- Lowering the mean by 0.05 gives a tie at 8.78, not a lead. A clear first place needs about -0.10 or more, so two or three ideas must stack.

## New measurements

All on hybrid B OOF (checked: 6.312 / 10.204 / 10.115, same as the brief), 8954 rows, project folds.

| Measure | CD | CV | HEX |
|---|---:|---:|---:|
| r within source A | 0.810 | 0.561 | 0.564 |
| r within source B | 0.885 | 0.539 | 0.467 |
| r, all rows | 0.862 | 0.695 | 0.605 |
| Fellow-eye label r, source B | 0.821 | 0.408 | 0.251 |
| Fellow-eye residual r, source B | 0.563 | 0.226 | 0.093 |
| Residual log SD, A / B | 0.101 / 0.089 | 0.137 / 0.132 | 0.127 / 0.129 |

Sources:

| Source | ID format | Years | Frames | Mean CD | Mean CV | Mean HEX | CD MAPE |
|---|---|---|---:|---:|---:|---:|---:|
| A | `NNNN-YY` | 2018-2024 | 4514 | 2927 | 0.418 | 0.515 | 7.11 |
| B | `YYYY-NNNN ODCN/OSCN` | 2019-2022 | 4440 | 2664 | 0.347 | 0.576 | 5.50 |

- The two formats overlap in time, so they are not old and new. The two sources differ by 0.07 in mean CV. About half of the all-rows CV r is source identification: within a source it is 0.54-0.56, against 0.88-0.90 from expert markers (25 frames, source A, 2023).
- Correction: the 0.563 CD residual r is inflated by a few frames with extreme residuals. Spearman 0.343, winsorised 0.404 (CV Spearman 0.18, HEX 0.10). CD residuals still share more between eyes than CV and HEX do, but by less than first stated.
- Regression to the mean explains about 0.2 of the CD figure (arithmetic from r, spread ratio and label SD). The same arithmetic gives about 0.38 for CV, above the measured 0.226. So CV and HEX residuals are mostly specific to the eye.
- Serial neighbours from different donors in source B share CD residual: lag 2 Pearson 0.189 but Spearman 0.091; lag 3 Spearman 0.050. Most of the apparent session effect in B is the same few frames.

Hidden pairs in source A:

- `slide_id` equals the ID in A, so `E.donors` makes every A frame a singleton. Consecutive serials look like pairs: label Spearman CD 0.49-0.53, CV 0.42, HEX 0.34-0.36; residual Spearman CD 0.43-0.45, CV 0.31-0.33. Median |d log CD| 0.06, random pair 0.116, true fellow eye in B 0.05. Same-fold share 0.187, so folds split them.
- A CD MAPE by neighbour status: neighbour in training 6.88 (n 3293, +-0.31); all neighbours held out with the frame 7.62 (n 484, +-0.77); no neighbour 7.85 (n 737, +-0.58). CV and HEX show no such pattern. The 6.88 vs 7.62 difference is not significant on its own. Singletons may differ for other reasons.
- Source B pairs stay together in the official table (29 singletons in 2257 slides), so the organisers appear to split on `slide_id`. If so, A pairs were not kept together in the test split. Unverified.

Ceilings with a link that is not available at test (serial number or fellow eye), other fold and other donor, nested strength:

| Oracle link | CD | CV | HEX | Mean | Change |
|---|---:|---:|---:|---:|---:|
| None | 6.312 | 10.204 | 10.115 | 8.877 | |
| Serial +-1..3 | 6.113 | 10.055 | 10.058 | 8.742 | -0.135 |
| Serial +-2..3 | 6.259 | 10.175 | 10.104 | 8.846 | -0.031 |
| Serial +-3..6 | 6.294 | 10.184 | 10.108 | 8.862 | -0.015 |
| Fellow eye, source B rows only | 4.972 | 10.053 | 10.133 | 8.386 (from 8.634) | -0.248 on B |

Already killed by these measurements:

- Per-source or per-year calibration: nested global log-affine 8.864, per source 8.866, per source-year 8.867.
- Linking frames by low-level style (illumination map, percentiles, radial spectrum; 274 features, PCA 32): true partner median rank 1198-1497 of about 4500, residual kNN r about 0, gain 0.
- Using the fellow eye at test in source B: both eyes appear to fall in the same split, and cases are assumed to run in isolation (assumption, not checked on the platform).

## Ranked ideas

Gains in the "size" column are arithmetic or assumption, not forecasts. All gates are on full labels, nested, paired against the current best OOF.

| # | Idea | Mechanism | Size | Cheapest kill test | Gate |
|---|---|---|---|---|---|
| 1 | CD density channel without the CNN | CD comes from the mean area of about 150 cells. Fourier ring spacing needs no peak detection, and detector density in fixed-count windows reads the same quantity with errors unlike the CNN's. Best pipeline CD is hybrid B plus stretch only. | CD -0.10 to -0.20 if the residual holds count information | CPU, 1-2 h, all frames: partial correlation of the features with hybrid B CD residual per source, then nested log-ridge | Kill above CD -0.05 or if the CI includes 0. Adopt at CD -0.15 or lower (mean -0.05). Between: keep for stacking. |
| 2 | Audit of the device rule on the 25 marker sets | Hull area r -0.93 with CD, yet CD read from markers r 0.75; label / marker density 0.97-1.43; device CV twice Voronoi CV. Either the formula is wrong or the markers are incomplete or not the device's. | None directly. It decides how far to trust the marker table behind 3 and 5. | CPU, 2 h: grid of tessellations, cell sets and rounding; gap fraction inside the hull against the density ratio | Pass: CV within 0.01 on at least 20 of 25, or r at least 0.97. Fail: marker r values are not a ceiling; lower 3 and 5. |
| 3 | Thinning-robust point statistics for CV | Random misses leave the pair correlation g(r) unchanged, while one missed cell breaks Voronoi area CV. CV residual is specific to the eye, so only frame geometry can help. Use g(r) shells and number variance from soft detector maps. | CV -0.10 or more if the marker table holds | CPU, 2 h: the 25-set table row miss 11.5 %, false 14 %, jitter 10 % (now r 0.76, MAPE 14.2), then all frames | Kill if r is below 0.80 on the row and no change on all frames. Adopt at CV -0.10 or lower, on top of the graph network. |
| 4 | Per-image MAPE shift | The MAPE optimum is exp(mu - sigma^2). The flat -0.02 bias matches one sigma of 0.138. Frames with a different sigma get the wrong shift. | Mean -0.02 to -0.03, assuming sigma ranges 0.08-0.25 (not measured) | CPU, 30 min: model the absolute residual from member disagreement and quality, apply nested | Kill above mean -0.015. |
| 5 | Peak repair from image evidence, using the 5-minute budget | Check Voronoi edges against dark ridges in the image; add or remove peaks where the evidence disagrees. Uses no regularity prior, so it does not flatten CV. Misses and false peaks cost the most in the marker table. | CV -0.10 or more if the detector is the limit | CPU first: on 25 sets, without using markers, miss + false from 0.315 to 0.25 or lower with jitter no worse | Kill on that 25-set test. Then full labels, CV -0.10 or lower; graph retrain needs GPU. |
| 6 | Fixed-count windows for the graph network | Technicians mark about 148 cells whatever the density, so the region area changes with CD; our windows do not. | CV + HEX -0.10 | CPU, 10 min: OOF CV and HEX error by CD tertile; flat means kill | CV + HEX -0.10 or lower. Graph rebuild CPU, retrain GPU. |
| 7 | Learned donor memory for source A | Oracle ceiling -0.135, mostly A's +-1 neighbours. Needs A test frames to have their partner in training (unverified), a link stronger than low-level style, and a rule check. | Ceiling times link precision | GPU, short: train a pair embedding on B's 2184 true pairs, test on A's neighbours | Kill below 40 % top-1 precision on A. |

## Separate issue for the PI

Source A neighbours fall in different folds, so OOF on A may reward memorising a donor. If the test split kept A pairs together, A test error would be worse than OOF. Test CD is below OOF (6.250 vs 6.312), which argues against that but does not settle it.

## Scripts

`clear_ec_ideation_scripts/` next to this file: `label_forensics2.py`, `resid_forensics2.py`, `session_ceiling.py`, `style_link.py`, `partner_fold.py`. Run on the pod from stdin; they write nothing there.
