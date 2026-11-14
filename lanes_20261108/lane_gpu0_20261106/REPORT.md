# Lane GPU 0 report (started 2026-11-06)

Gates are in `protocol.json`. Sources in `source/`. Reads in `reads/`. Times Eastern.

## Reads so far

| Read | Change on the mean | CI | CD | CV | HEX | Gate | Verdict |
|---|---:|---|---:|---:|---:|---|---|
| A1, oracle: -22 rows get their own bias (IDs, not available at test) | +0.003 | -0.007, +0.013 | +0.004 | +0.004 | 0.000 | none, ceiling | -22 has no shared bias |
| A1, oracle: serial 173-550 of -22 gets its own bias (IDs) | -0.013 | -0.024, -0.001 | -0.001 | -0.021 | -0.017 | none, ceiling | Ceiling just above the correction gate |
| A1, image only: residual on the detector score for that window | 0.000 | -0.005, +0.005 | +0.001 | +0.003 | -0.004 | -0.01 | Fail |
| A1, image only: residual on the -22 detector score | +0.001 | 0.000, +0.003 | +0.002 | +0.002 | +0.001 | -0.01 | Fail |
| A1, image only: gradient boosting on 90 label-free frame statistics | +0.059 | +0.041, +0.075 | +0.018 | +0.088 | +0.069 | -0.01 | Fail, memorises |
| Oracle: all rows get their serial-order bin's bias, 60-row bins, shrink 60 (IDs) | -0.019 | -0.035, -0.003 | -0.040 | -0.014 | -0.004 | none, ceiling | Whole-acquisition-order ceiling |
| B 2019 hyphen rows, oracle: own median log-affine map (IDs) | -0.010 | -0.027, +0.003 | -0.017 | -0.017 | +0.003 | none, ceiling | CI crosses 0 even with IDs |
| B 2019 hyphen rows, image only: score above 0.5 gets the map | 0.000 | -0.001, +0.001 | 0.000 | +0.001 | -0.001 | -0.01 | Fail |

## Measured augmentation, encoder training (fold 4 running)

Last five epochs, paired with the matched two-rank reference. One fold decides nothing; the gate needs folds 0 and 4.

| Read | Change on the mean | CI | CD | CV | HEX |
|---|---:|---|---:|---:|---:|
| Fold 0, seed 46, member | +0.004 | -0.043, +0.049 | +0.024 | -0.014 | +0.001 |
| Fold 0, added to best at weight 0.25 | +0.002 | -0.015, +0.018 | +0.006 | +0.008 | -0.009 |
| Fold 0, added at 0.125 (not the gate) | -0.001 | -0.010, +0.006 | +0.001 | +0.002 | -0.008 |

Arm 9.084 against reference 9.081. Train probe at epoch 20: 8.359.

## RoPE coordinate scale at inference: closed at the pipeline

The encoder draws a random coordinate rescale in [1/2, 2] at every train-mode forward (DINOv3 hub default, not in the audit) and validates at 1. A fixed scale at inference needs no retraining.

| Read, same stretch on both arms | Change on the mean | CI | CD | CV | HEX |
|---|---:|---|---:|---:|---:|
| Seed 46, folds 0 and 4, member, scale 1.414 (first look) | -0.071 | -0.103, -0.040 | -0.044 | -0.111 | -0.060 |
| Seed 46, five folds, member | -0.032 | -0.052, -0.013 | -0.002 | -0.021 | -0.074 |
| Seed 47, five folds, member | -0.017 | -0.036, +0.002 | +0.037 | -0.021 | -0.068 |
| Seeds 46 and 47, five folds, member | -0.022 | -0.040, -0.003 | +0.022 | -0.018 | -0.068 |
| Hybrid B + stretch | -0.001 | -0.010, +0.009 | +0.009 | +0.003 | -0.015 |
| **Best pipeline, against best** | **+0.002** | **-0.007, +0.011** | +0.009 | +0.009 | -0.011 |

All scales on all members, seeds 46 and 47, five folds:

| Scale | Member | Member CI | Best pipeline | Pipeline CI | Pipeline CD | Pipeline CV | Pipeline HEX |
|---|---:|---|---:|---|---:|---:|---:|
| 1.19 | -0.020 | -0.030, -0.009 | -0.001 | -0.007, +0.004 | -0.002 | +0.003 | -0.005 |
| 1.414 | -0.022 | -0.040, -0.003 | +0.002 | -0.007, +0.011 | +0.009 | +0.009 | -0.011 |
| 1.68 | -0.009 | -0.034, +0.015 | +0.007 | -0.005, +0.018 | +0.021 | +0.018 | -0.019 |
| Chosen on the other folds (1.19 in all five) | | | -0.001 | -0.007, +0.004 | -0.002 | +0.003 | -0.005 |

The gain at member level is mostly HEX, which hex6 already takes, and CD gets worse as the scale grows. The first look was one seed on two folds and overstated it threefold. Fails the correction gate; closed. Curve on the first look: 0.707 +0.202, 1.19 -0.052, 1.68 -0.061, 2.0 +0.025, 2.38 +0.830; anisotropic scales no better.

Open: training with the rescale off (`rope_fixed` in the queue) tests a different thing, whether the noise itself costs anything.

## Noise, focus and exposure of real frames

All 8954 raw frames, before CLAHE. `measure_frames_20261106.py`, `calib2_20261106.py`.

Calibration on 120 frames with known blur and noise:

- White noise floor from the spectrum, 0.27-0.40 cycles/px, reads added noise back within 0.02: 3.383 against 3.37 expected at sigma 2, 4.846 against 4.833 at 4, 6.583 against 6.585 at 6.
- Focus, as the spectral roll-off over 0.07-0.14 cycles/px with the floor removed: blur 1 adds 0.64, 1.5 adds 1.88, 2 adds 3.63, refill or not. Blur 0.5 is not resolved. Added noise moves it by under 0.01. My first estimator, band 0.08-0.25, went the wrong way and is not used.

| Measure | Source | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Noise floor sigma, grey levels | A | 2.43 | 2.59 | 2.81 | 3.58 | 3.79 | 4.02 | 4.29 |
| | B | 2.00 | 2.14 | 2.38 | 2.53 | 2.67 | 2.80 | 2.92 |
| Focus | A | 3.71 | 4.23 | 4.94 | 5.45 | 6.04 | 7.05 | 7.74 |
| | B | 3.80 | 4.27 | 4.95 | 5.43 | 5.98 | 6.86 | 7.56 |
| Median grey level | A | 12 | 30 | 60 | 81 | 102 | 130 | 149 |
| | B | 65 | 83 | 106 | 123 | 138 | 155 | 166 |
| p99 grey level | A | 109 | 139 | 176 | 202 | 224 | 248 | 255 |
| | B | 118 | 142 | 176 | 201 | 220 | 237 | 248 |
| Share of black pixels | A | 0 | 0.004 | 0.042 | 0.093 | 0.170 | 0.317 | 0.427 |
| | B | 0 | 0 | 0 | 0 | 0 | 0 | 0.007 |

What this says:

- No real frame has less grain than sigma 2. Blur alone leaves the real range at once: floor 1.83 at sigma 0.5, 0.46 at sigma 1. The noise range in the earlier bundle, 1 to 7 added, goes past the real maximum.
- Focus spread is about the same in both sources. p5 to p95 equals added blur of about sigma 1.7, the quartiles about 1.2. Some of that may be cell structure rather than optics.
- Noise grows with intensity: exponent about 0.4 in A (clipped darks), 0.95 in B.
- JPEG-like traces in all frames: spectral bumps at 0.25 and 0.5 cycles/px and an 8-px grid (ratio 1.05-1.09).
- Grey-level count marks processing eras. A has mostly 120 levels up to 2021 and mostly 166 from 2023; the switch is inside 2022, around serial 550-700 of `-22`. B falls slowly, 240 in 2019 to 204 in 2022.

Link to error: within each source, no frame statistic reaches Spearman 0.06 with the signed log residual of the best pipeline. The largest with absolute residual is -0.10 (vignette, B, CD). Real variation in noise, focus and exposure does not by itself explain the error.

## Response of a trained member

Fold 0 seed 46 reference, epoch 20, 240 validation frames. `response_test_20261106.py`. Change in log prediction, median, percent; MAPE on the same frames.

| Perturbation | CD | CV | HEX | MAPE mean |
|---|---:|---:|---:|---:|
| None | | | | 8.887 |
| Blur 1, no refill (earlier test) | +1.1 | +5.9 | -5.3 | 11.004 |
| Blur 2.5, no refill (earlier test) | -0.6 | +12.2 | -10.0 | 13.011 |
| Defocus 1, grain refilled | -2.3 | -1.9 | +0.8 | 9.655 |
| Defocus 1.5, grain refilled | -2.9 | -1.5 | +0.6 | 9.820 |
| Noise sigma 6 (earlier test) | -5.6 | -5.5 | +2.8 | 10.926 |
| Noise up to the source's p99 floor | -1.6 | -2.9 | +2.3 | 9.088 |
| Gain 0.7 | -0.2 | -0.1 | +0.1 | 8.852 |
| Gain 1.3 | 0.0 | +0.7 | -0.6 | 9.092 |
| Measured transfer, random same-source donor | 0.0 | 0.0 | +0.2 | 9.362 |

- The earlier "blur sigma 1 moves CV by +6.8 %" reproduces, but it is mostly removal of grain. With the grain put back, CV moves -1.9 %, the other way. The model reads grain.
- Inside the real range the median shift is near zero, but single frames move (median absolute 0.9, 1.2, 1.0 %) and MAPE on these frames rises by 0.48. This is why measured augmentation is still worth a run. The transfer only degrades frames, so part of the 0.48 is lost information rather than sensitivity.

## Measured augmentation

`recipeA_measured_aug2_20261106.py`. With probability 0.8, pick a donor from the same source among the fold's training rows, then in acquisition order: blur up to the donor's focus if it is softer; match its median and p99 linearly, gain within 0.6 to 1.6 and capped so own grain stays within the larger floor; add signal-dependent white noise up to the donor's floor. Clipped black stays black. No rotation. Flips as before.

Check on 600 frames, augmented against real, p5 / p50 / p95:

| Measure | Source | Real | Augmented |
|---|---|---|---|
| Noise floor | A | 2.59 / 3.58 / 4.02 | 2.64 / 3.70 / 4.25 |
| | B | 2.14 / 2.53 / 2.80 | 2.15 / 2.53 / 2.81 |
| Focus | A | 4.23 / 5.45 / 7.05 | 4.87 / 5.93 / 7.21 |
| | B | 4.27 / 5.43 / 6.86 | 4.81 / 5.94 / 7.28 |
| Median grey | A | 30 / 81 / 130 | 23 / 79 / 129 |
| | B | 83 / 123 / 155 | 82 / 121 / 155 |
| p99 grey | A | 139 / 202 / 248 | 137 / 194 / 240 |
| | B | 142 / 201 / 237 | 137 / 197 / 234 |

Focus shifts softer by about 0.5 because the transfer never sharpens.

## One-GPU runner

`recipeA_single_gpu_20261106.py` runs the frozen two-rank runner in one process, 4 micro-batches of 4, the same grouping and the same random draws as the two ranks. Micro-batch 8 failed the frozen smoke gate (relative gradient error 0.0023 against 0.002), so it was dropped. A first version made 4 RoPE draws per update instead of 2 and differed at update 1; fixed. 5.4 frames per second, about 26 minutes per epoch, about 8.7 hours per fold.

Control, fold 0 seed 46, one epoch, flips only: gradient norm against the two-rank log 0.54351 / 0.5435 at update 1, ratio 1.000-1.005 through update 125, then the paths separate numerically. Epoch 1 mean 10.250 against 10.332, r log 0.974-0.981, so the gate I wrote failed. I judged the gate too tight, because epoch 1 ranges 9.93-10.63 across seeds, and did not train the 20-epoch one-process reference it called for. Last-five-epoch means across seeds differ by at most 0.017 (fold 0) and 0.009 (fold 4).

## Audit A1: `-22` rows

Best pipeline OOF by year, mean of the three metrics: -18 8.57, -19 9.00, -20 8.66, -21 8.80, **-22 9.91**, -23 8.85, -24 8.93; B 2019 9.21, B 2020 7.90, B 2021 8.15, B 2022 8.50. If `-22` rows had the error of other A rows the overall mean would drop by 0.083. That is what a perfect fix would give, not what a correction can reach.

What differs:

- `-22` straddles a processing change. Serial 1-550 looks like earlier years (120 grey levels, high-pass SD 10-12). From about 550 it looks like 2023-24 (136-166 levels, high-pass SD 6-8).
- Serial 173-550 (189 rows) stands out: mean 9.2-12.5 per bin, CV bias -0.06 to -0.11, HEX bias +0.04 to +0.06, labels CV 0.43-0.47 against 0.42, HEX 0.48-0.50 against 0.51. The frames show no matching change in grey levels or contrast.
- Overall `-22` has no shared bias, so a `-22` correction has nothing to remove. Its excess is spread: residual SD 0.110, 0.145, 0.135 against about 0.10, 0.135, 0.12.

Correction: fails at every level above. Even with IDs the window gives only -0.013. From the image the window has AUC 0.77 within A, 0.68 among frames with the same grey-level count, and gives 0.000. A1 is closed for corrections.

A subgroup larger than `-22`, found while scanning acquisition order: 523 rows of B 2019 have IDs spelled with a hyphen (`2019-0064-OSCN`), against 659 spelled with a space. Serial 2-786, 272 donors, 12 of them with both spellings.

| Rows | n | CD MAPE | CV MAPE | HEX MAPE | CV, top 1 % removed | r log CV | r log CD | r log HEX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B 2019 hyphen | 523 | 5.32 | 16.53 | 10.75 | 12.89 | 0.19 | 0.95 | 0.45 |
| B 2019 space | 659 | 4.36 | 9.55 | 9.78 | 9.08 | 0.53 | 0.96 | 0.45 |
| B 2020-22 space | 3258 | 5.4-6.0 | 8.7-9.8 | 9.0-10.1 | 8.92 | 0.58 | 0.85 | 0.55 |

CD and HEX are read as well as elsewhere; only CV is nearly unreadable on these rows, so their CV labels look noisy or differently derived. They are also darker (median grey 116 against 140, d -1.24). Correction: with IDs a median log-affine map gives -0.010 (CI -0.027, +0.003), CD -0.017, CV -0.017, HEX +0.003. From the image, hyphen score AUC 0.73 within B 2019, 0.81 within B, and the score-gated map gives 0.000. Fails the gate. Not tested: leaving these rows' CV out of the training loss, which is possible because IDs are known at training time.

Extreme rows, found on the way, worth knowing for any read of OOF:

- The tail is mostly labels the image contradicts. One session, B 2020 serial 644-651, has 6 frames with CD labels 372-926 where the model says 1560-3010, both eyes of each donor. Together they hold about 0.19 on CD, 0.06 on the mean. The two rows with HEX 0, `2019-0064`, also have CV 0.03 and 0.11; CV APE 1375 and 309 hold 0.19 on CV, 0.06 on the mean.
- With APE clipped at 50 the mean is 8.568, at 100 8.629, against 8.739. Top 10 rows hold 4.3 % of CD error, 2.6 % of CV. On 1000 test rows these rows land by chance, which adds to the test-against-OOF gap and to any fold's noise.
- Label-clean training (dropping 3 %) was already closed at +0.0003, so this is a fact about the readout, not a proposal.
