# Lane GPU 2: pooling and heads, within-donor training, then strong-view SSL at a second seed

Plan rows G3 and G5. Gates and every amendment are in `protocol.json`. Sources in `source/`, reads in `reads/`. Times Eastern.

## Where things stand, 2026-11-07 20:15

- Pooling and heads: closed on this encoder. No arm passed the frozen screen, and the check on the screen's known bias came out below its limit.
- Corrections from token statistics on top of the best pipeline: none passed.
- Within-donor training: fold 0 reads +0.103 (CI +0.050, +0.154) against its reference, and the difference between fellow eyes is read no better. Fold 4 was cancelled on the coordinating session's instruction.
- Strong-view SSL encoder, end to end at regression seed 47: fold 4 reads +0.003 (CI -0.046, +0.055) against its reference. Lane GPU 1's seed 46 failed both gates on two folds, so fold 0 seed 47 was stopped at 07:38 on 11-07: it could not change the verdict.
- DINOv3 ViT-B/16 at stride 16, two seeds: gate missed. Added to the best pipeline at w 0.125 it reads -0.0166 (CI -0.0282, -0.0053) against a gate of -0.02. Seed 47 repeats seed 46, and neither a second seed nor an earlier stopping point moves it to the gate.
- DINOv3 at stride 8, lane GPU 3's arm: fold 0 reads worse than stride 16 on the same fold, both as a member (+0.471 against +0.294) and added to the best pipeline (w 0.125 -0.0054 against -0.0156). One fold, information only. Fold 4 started at 17:28 on 11-07 and ends about 06:30 on 11-08; the two-fold read under lane GPU 3's gates follows.

## Frozen read, each fold's own encoder (primary)

Head as recipe A on the fold's training rows, read on its validation rows, three head seeds, mean of epochs 16-20. Folds 0 and 4 pooled, paired donor bootstrap against std, which reads 9.0417. Std against the trained member's epochs 16-20 mean: fold 0 -0.017, fold 4 +0.008, limit 0.10, so the read is valid.

Screen for end-to-end training: change <= -0.03, CI below 0, both folds negative. "Added" is the arm's difference from std added to the best pipeline at weight 0.25, on these two folds; the plan's gate for a new member is <= -0.02.

| Arm | Change | CI | CD | CV | HEX | Fold 0 | Fold 4 | Added | Added CI |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Quantile head | -0.029 | -0.044, -0.013 | +0.011 | -0.063 | -0.036 | -0.023 | -0.035 | +0.0002 | -0.0032, +0.0038 |
| Std + radial spectrum of 64 PCs | +0.005 | -0.028, +0.037 | +0.016 | -0.010 | +0.009 | +0.030 | -0.020 | -0.0091 | -0.0187, -0.0001 |
| Std + radial autocorrelation | +0.014 | -0.014, +0.041 | +0.014 | +0.010 | +0.018 | +0.012 | +0.016 | -0.0095 | -0.0165, -0.0025 |
| Std + covariance square root | +0.019 | -0.002, +0.041 | -0.017 | +0.028 | +0.047 | +0.012 | +0.027 | -0.0085 | -0.0140, -0.0026 |
| Std + local heterogeneity | +0.022 | +0.000, +0.042 | +0.038 | +0.018 | +0.008 | +0.025 | +0.018 | +0.0014 | -0.0042, +0.0069 |
| Mean + std | +0.024 | +0.010, +0.037 | +0.013 | +0.029 | +0.030 | +0.020 | +0.027 | +0.0007 | -0.0027, +0.0041 |
| Std + autocorrelation + covariance square root | +0.025 | -0.002, +0.053 | +0.014 | +0.019 | +0.041 | +0.029 | +0.021 | -0.0094 | -0.0161, -0.0025 |
| Std + autocorrelation + spectrum | +0.027 | -0.003, +0.056 | +0.002 | +0.030 | +0.049 | +0.035 | +0.020 | -0.0105 | -0.0181, -0.0026 |
| Std + per-window head outputs | +0.030 | +0.005, +0.054 | +0.025 | +0.025 | +0.040 | +0.034 | +0.026 | +0.0003 | -0.0066, +0.0068 |
| Std + multiscale std | +0.032 | +0.015, +0.050 | +0.032 | +0.030 | +0.034 | +0.029 | +0.035 | -0.0011 | -0.0056, +0.0035 |
| Std + covariance | +0.036 | +0.012, +0.058 | -0.005 | +0.061 | +0.051 | +0.025 | +0.046 | -0.0058 | -0.0113, +0.0001 |
| HL-Gauss head | +0.036 | +0.003, +0.071 | +0.161 | -0.048 | -0.004 | +0.026 | +0.046 | +0.0118 | +0.0025, +0.0221 |
| Std + quantiles | +0.038 | +0.017, +0.059 | +0.037 | +0.023 | +0.055 | +0.041 | +0.035 | +0.0032 | -0.0026, +0.0087 |
| All statistics | +0.040 | +0.009, +0.072 | +0.041 | +0.039 | +0.042 | +0.079 | +0.002 | -0.0110 | -0.0190, -0.0029 |
| Std + skew + kurtosis | +0.058 | +0.033, +0.084 | +0.006 | +0.070 | +0.098 | +0.065 | +0.051 | +0.0047 | -0.0020, +0.0118 |
| Std + class token | +0.065 | +0.034, +0.097 | +0.093 | +0.077 | +0.024 | +0.096 | +0.034 | +0.0021 | -0.0067, +0.0100 |

Reading:

- The quantile head misses the screen by 0.001. The gate was not moved. Added to the best pipeline it gives nothing, so a five-fold correction read was not run. Its gain is probably the calibration that the stretch already provides; this was not tested separately.
- The best added read is -0.011, about half the gate, and its weight is the best of three seen after the fact.
- The two combinations were added after the first results were seen, as a filter for the end-to-end choice. That choice did not arise.

Cross-fit read (secondary): each row from the encoder that held it out. Its std reads 0.11 and 0.13 worse than the members, which missed the 0.10 limit, so it cannot decide the screen. Its spectrum arm reads -0.102 (CI -0.146, -0.057) and its quantile head -0.037 (CI -0.055, -0.019). The spectrum figure is not a gain: the cross-fit reference is weakened by misalignment between encoders and the spectrum is less affected. On the primary read the same arm is +0.005.

## How far the frozen read can be trusted

The screen's known bias is that these encoders were trained for std. It had never been measured. The rule, written before the read and after the table above was seen: if the frozen gap of mean against std exceeds the trained gap by 0.08 or more, one arm goes end to end despite the screen.

| Mean against std, last five epochs | Frozen | Trained |
|---|---:|---:|
| Fold 0 | +0.089 | +0.120 (seed 47; CI +0.061, +0.182) |
| Fold 4 | +0.155 | +0.049 (seed 46; CI -0.013, +0.110), +0.051 (seed 47; CI -0.009, +0.114) |
| Pooled, one seed per fold | +0.122 | +0.085 |

The difference is 0.037, below the limit. On fold 4 alone it is 0.106; on fold 0 it is -0.031. Mean + std agrees on fold 4: frozen +0.027, trained seed 46 +0.032. So the screen stands, with this limit: it is one measured pair, and mean is a different statistic from the ones in the table.

Side result: on fold 4 seed 46, last five epochs, trained arms read mean +0.049, mean + std +0.032, max +0.052, min +0.057, mean + std + min + max +0.046 against std. Each CI includes 0 on this one fold. The earlier terminal-epoch figure for mean, +0.085, was larger.

## Corrections from token statistics, five folds on the best pipeline (secondary)

Ridge on whitened PCs of each block, cross-fit features, settings chosen in inner folds. Gate: <= -0.01, CI below 0, 4 of 5 folds.

| Block | Change | CI | Block | Change | CI |
|---|---:|---|---|---:|---|
| Legible-window head output | -0.0024 | -0.0064, +0.0014 | Local heterogeneity | +0.0055 | -0.0056, +0.0169 |
| Per-window head statistics | -0.0019 | -0.0105, +0.0067 | Kurtosis | +0.0070 | -0.0042, +0.0173 |
| Spectrum | +0.0003 | -0.0103, +0.0107 | Mean | +0.0092 | -0.0026, +0.0218 |
| Quantiles | +0.0009 | -0.0101, +0.0115 | Covariance square root | +0.0110 | -0.0008, +0.0235 |
| Covariance | +0.0015 | -0.0089, +0.0125 | Class token | +0.0113 | +0.0015, +0.0205 |
| Autocorrelation | +0.0042 | -0.0074, +0.0165 | Legible-window std | +0.0130 | +0.0023, +0.0233 |
| Std | +0.0052 | -0.0058, +0.0158 | Multiscale std | +0.0154 | +0.0059, +0.0254 |
| | | | Skew | +0.0178 | +0.0072, +0.0289 |

None passes.

Exploratory: most blocks lower HEX while raising CD. Applying the same corrections to HEX only gives mean -0.0126 (CI -0.0205, -0.0045; folds 0.000, -0.009, -0.016, -0.016, -0.022), quantiles -0.0122, std -0.0116, local heterogeneity -0.0100. These numbers clear the gate, but the metric was chosen after seeing the outer folds, so they do not pass it. Submission 1's HEX was also the metric furthest above its OOF on the test set (+0.395). Corrections are in `reads/v2/corr/`, results in `hexonly.jsonl`.

## Within-donor training, fold 0

Recipe A with partners adjacent in each chunk and a loss on the error difference, lam 1, seed 46. The old crash is fixed and the run passed its audit. Read against `phase2_pooling_replication_20261026/f0_s46_std`, mean of epochs 16-20, 1791 rows.

| | Within-donor training | Reference | Change |
|---|---:|---:|---:|
| Mean | 9.1835 | 9.0807 | +0.103 (CI +0.050, +0.154) |
| CD | 6.475 | 6.267 | +0.208 |
| CV | 10.483 | 10.397 | +0.086 |
| HEX | 10.593 | 10.578 | +0.015 |

Added to the best pipeline on these rows: w 0.125 +0.0005 (CI -0.0085, +0.0093), w 0.25 +0.0071 (CI -0.0107, +0.0247). Neither reaches the gate of -0.02.

The idea's own target did not move either. Over 436 fellow-eye pairs, r between the label difference and the predicted difference, against the reference:

| Metric | Within-donor training | Reference | Change | CI |
|---|---:|---:|---:|---|
| CD | 0.723 | 0.737 | -0.014 | -0.035, +0.004 |
| CV | 0.272 | 0.285 | -0.014 | -0.049, +0.023 |
| HEX | 0.160 | 0.187 | -0.027 | -0.069, +0.011 |

Limit: this is one fold and one seed, which the plan's rule says decides nothing. Fold 4 was cancelled by the coordinating session in favour of the strong-view SSL run, so this idea is stopped, not closed by the rule. The fold 0 result gives no reason to resume it. Chunk order also changed in this run, which the gain from pairing would have had to outweigh; the effect of order alone was not measured.

## Strong-view SSL encoder, regression seed 47

Arm `cropopt_both` from lane GPU 1: crop + broad optics views, SSL pools by mean and std, SSL seed 1003. Recipe A on one GPU, std pooling, 20 epochs. Reference: `phase2_pooling_replication_20261026/f{4,0}_s47_std`, same regression seed on the SSL 5 encoder. Pass rule as the plan.

- Roots: `e2e/cropopt_both_f{4,0}/f{4,0}_s47`. Lane GPU 1's own seed 46 runs are in its folder.
- Runners: fold 4 `source/run_e2e_gpu2_20261107.sh`, fold 0 `source/run_e2e_gpu2_v3_20261107.sh`. Each is a copy of lane GPU 1's runner with the root moved here. The second does not rewrite `regression_input` once it is newer than the finished SSL, so it cannot disturb lane GPU 1's running audit. It passed through smoke on 11-06 at 23:26 on the finished SSL (5 epochs, sha ddcf1e0e), and the job reuses that manifest. `source/run_e2e_gpu2_v2_20261107.sh` is not used: it has lane GPU 1's quoting bug.
- Fold 4 ran 22:57 on 11-06 to 06:57 on 11-07, 8.0 h, and passed its audit. Fold 0 started at 06:57 and was stopped by pid at 07:38, with one epoch complete. Its root `e2e/cropopt_both_f0/f0_s47` stays, has no `exit.json` and holds no read.
- Why it was stopped: lane GPU 1's seed 46 reads +0.0185 on two folds (CI -0.020, +0.058), added w 0.125 -0.0038 (CI -0.0097, +0.0017), residual r with recipe A 0.98-0.99. Checked in its `reads/e2e_both_s46.json`. A pass at seed 47 would have needed about -0.10 on fold 0.

Fold 4, mean of epochs 16-20, 1790 rows:

| | Strong-view SSL | SSL 5 reference | Change |
|---|---:|---:|---:|
| Mean | 9.0112 | 9.0080 | +0.003 (CI -0.046, +0.055) |
| CD | 6.652 | 6.605 | +0.047 |
| CV | 10.244 | 10.216 | +0.029 |
| HEX | 10.137 | 10.203 | -0.066 |

Added to the best pipeline on these rows: w 0.125 -0.006 (CI -0.014, +0.002), w 0.25 -0.007 (CI -0.022, +0.009). The plan's gate is <= -0.02 with CI below 0.

Reading:

- This fold alone decides nothing, and the arm's failure rests on lane GPU 1's two folds at seed 46. For a two-fold mean at seed 47 to reach -0.05, fold 0 would have had to read about -0.10.
- The frozen gain did not carry over on this fold. Lane GPU 1's frozen read of this arm on fold 4 is -0.166 (CD -0.426); trained through, the mean is +0.003 and CD +0.047. The two reads use different heads, so the figures are not strictly comparable. That frozen reads shrink under fine-tuning was already in lane GPU 1's calibration.
- The run is valid, although its curve looks like the reference's (epochs 16-20: 9.008, 9.024, 8.995, 9.035, 8.995 against 9.004, 9.028, 8.982, 9.034, 8.993; best epoch 8.952 against 8.953). It started from a different SSL checkpoint and encoder tensor (2a3fc29e, 619e95ba against 2b046c0c, b294cf85), with the same head start from the shared seed.
- The trained model does differ from the reference, more than a second seed does. SD of the log difference on CD, CV, HEX: 0.017, 0.020, 0.015 against the reference, while seeds 46 and 47 of the reference differ by 0.012, 0.011, 0.009. So these views change what is learned without changing the score on this fold. Training rows read 8.609 against 8.548.
- The epoch-to-epoch movement is largely shared by the two runs. That suggests part of the within-run noise comes from the seed and data order, which a same-seed pair cancels. Measured on this one pair only.

## DINOv3 ViT-B/16 at a second seed: gate missed

Gate, written before running, in `protocol.json`: added <= -0.02 with CI below 0, at w 0.125 or 0.25, for the two seeds averaged in log. Seed 47 must have the same sign as seed 46 at that weight.

The gate is missed by 0.0034 at w 0.125 and by 0.0053 at w 0.25, where the CI also includes 0. The gate was not moved, so this arm is not a candidate.

Folds 0 and 4, 3581 rows, mean of epochs 16-20. Added figures are against the best pipeline, which reads 8.7578 on these rows.

| | Member against recipe A | Added, w 0.125 | Added, w 0.25 |
|---|---|---|---|
| Seed 46 | +0.274 (CI +0.185, +0.361) | -0.0165 (CI -0.0284, -0.0043) | -0.0123 (CI -0.0363, +0.0118) |
| Seed 47 | +0.284 (CI +0.194, +0.372) | -0.0156 (CI -0.0277, -0.0043) | -0.0098 (CI -0.0339, +0.0125) |
| Two seeds, averaged in log | not read | -0.0166 (CI -0.0282, -0.0053) | -0.0147 (CI -0.0378, +0.0076) |

Two seeds at w 0.125 by metric: CD -0.0178, CV -0.0253, HEX -0.0068. At w 0.5 they read +0.0484 (CI +0.0026, +0.0933), so the gain does not grow with weight. Seed 47's per-epoch form at w 0.125 is -0.015 (CI -0.0271, -0.0038), which agrees with its last-five read.

Reading:

- The small gain repeats. A new regression seed gives the same sign and nearly the same size. This replicates over training seed only: both seeds share the same rows, the same SSL encoder and the same pipeline, so the two CIs are not independent evidence about the data.
- A second seed does not help at w 0.125 (-0.0166 against -0.0165 and -0.0156), and helps slightly at w 0.25. The seeds' log predictions correlate at 0.99, 0.98 and 0.97 on CD, CV and HEX.
- Residual r with recipe A at seed 47 is 0.91-0.95, as at seed 46. That is lower than the 0.98-0.99 of the recipe A variants, which is the likely reason this arm adds anything at all. This was not tested separately.

Exploratory, by epoch window. The windows were chosen after the curves were seen, so nothing here can pass a gate. On all four runs validation bottoms out at epochs 9-13, and the last-five mean sits 0.08-0.15 above that minimum, while training rows read 7.5-7.7. Recipe A TTA reads 8.978 on these rows.

| Epochs | Member, two seeds | Added w 0.125 | Added w 0.25 |
|---|---:|---|---|
| 5-9 | 9.210 | -0.0107 (CI -0.0213, -0.0005) | -0.0061 (CI -0.0275, +0.0145) |
| 7-11 | 9.172 | -0.0147 (CI -0.0250, -0.0048) | -0.0114 (CI -0.0315, +0.0080) |
| 9-13 | 9.171 | -0.0156 (CI -0.0260, -0.0055) | -0.0140 (CI -0.0346, +0.0062) |
| 11-15 | 9.184 | -0.0160 (CI -0.0268, -0.0054) | -0.0143 (CI -0.0354, +0.0066) |
| 13-17 | 9.201 | -0.0162 (CI -0.0271, -0.0052) | -0.0147 (CI -0.0362, +0.0069) |
| 16-20 | 9.236 | -0.0166 (CI -0.0282, -0.0053) | -0.0147 (CI -0.0378, +0.0076) |

Stopping earlier improves the member by 0.065 and leaves the added read unchanged, so a shorter schedule is not worth a run. No window reaches -0.02. This says nothing direct about stride 8, which changes the model rather than the stopping point, but a better DINOv3 member can no longer be assumed to add more. Figures are in `reads/dinov3_b16_epoch_windows.json`.

Runs and reads:

- Runner: lane GPU 3's, unchanged, `--seed 47`, SSL reused from seed 46 (the SSL seed is fixed). About 1 h per fold; fold 0 and fold 4 ended at 08:37 and 09:36 on 11-07. Roots `runs/dinov3_b16_f{0,4}_s47`.
- `reads/dinov3_b16_s47.json` is by lane GPU 3's read tool, against `f{0,4}_s47_std`. `reads/dinov3_b16_two_seeds.json` is by `source/read_two_seeds_20261107.py`, whose seed 46 figures match lane GPU 3's stored read to four decimals.
- Lane GPU 3's protocol sends only passes to seed 47, so this was an added test.

## DINOv3 ViT-B/16 at stride 8 (running, lane GPU 3's arm)

Lane GPU 3 handed over its two stride 8 runs and took them out of its queue. They test whether DINOv3 at recipe A's token density is a better member than stride 16. Runner unchanged, roots in lane GPU 3's folder, gates in its `protocol.json`.

| Job | Eastern | Result |
|---|---|---|
| Fold 0, from saved SSL | 09:37 to 17:28 on 11-07, 7.9 h | `reads/dinov3_b16_s8_s46_f0_only.json` here, for information only |
| Fold 4, SSL then regression, about 13 h | Started 17:28 on 11-07, ends about 06:30 on 11-08 | `lane_gpu3_20261106/reads/dinov3_b16_s8_s46.json`, the two-fold read |

Fold 0, seed 46, mean of epochs 16-20, 1791 rows. Reference `phase2_pooling_replication_20261026/f0_s46_std`, which reads 9.0807; the best pipeline reads 8.7648 on these rows. Stride 16 on the same fold and seed, read with the same tool, is beside it (`reads/dinov3_b16_s46_f0_only.json`).

| | Stride 8 | Stride 16 |
|---|---|---|
| Member | 9.5515 | 9.375 |
| Member against reference | +0.471 (CI +0.348, +0.608) | +0.294 (CI +0.166, +0.425) |
| CD, CV, HEX against reference | +0.647, +0.419, +0.346 | +0.341, +0.254, +0.288 |
| Added, w 0.125 | -0.0054 (CI -0.0221, +0.0116) | -0.0156 (CI -0.0338, +0.0012) |
| Added, w 0.25 | +0.0121 (CI -0.0212, +0.0466) | -0.0112 (CI -0.0472, +0.0222) |
| Added, w 0.5 | +0.1287 (CI +0.0639, +0.1965) | +0.0696 (CI -0.0004, +0.1359) |
| Residual r with reference, CD, CV, HEX | 0.885, 0.943, 0.952 | 0.907, 0.943, 0.951 |
| Training rows | 8.147 | 7.513 |
| Seconds per epoch | 1413 | 175 |

Reading:

- One fold and one seed decide nothing under the plan's rule. On this fold, stride 8 is worse than stride 16 on every line that matters, and costs eight times as much per epoch.
- Validation bottoms out at epoch 12 (9.452) and the last-five mean is about 0.10 above that, as with stride 16. Training rows read worse than stride 16 (8.147 against 7.513), so stride 8 fits less, not more.
- For the two-fold added read to reach -0.02 at w 0.125, fold 4 would need about -0.035. Stride 16 gave about -0.017 on fold 4.

## Mistakes and leftovers

- A local waiter used `timeout`, which this Mac does not have, so finished reads sat unread from 15:50 to 22:35 on 11-06. GPU 2 kept training throughout.
- The new queue fell through to fold 0 when fold 4 failed on lane GPU 1's test root, and started from that lane's SSL at epoch 1 of 5. Stopped after one update. It also wrote `ssl/cropopt_both_f0/regression_input` from that checkpoint; lane GPU 1's v2 runner rebuilds it unless it is newer than the finished SSL. The fold 0 runner now refuses an arm whose SSL has no `complete.json`.
- Early on, `du` was run on one 7.4 GB folder under `/workspace`, which is the Lustre mount and against the repo rule. Not repeated.

Left in place for the PI to delete or keep; none holds a result to use:

- `runs/cotwin_f0_s46`: prepare and CPU tests only, replaced by `runs/cotwin_mb4_f0_s46`. Its removal was blocked by the safety check.
- `lane_gpu1_20261106/e2e/cropopt_both_f0/INVALID_partial_ssl_f0_s47_20261107`: the stopped run above.
- `lane_gpu1_20261106/e2e/cropopt_both_f4/f4_s47/exit.json`: failure record of my attempt on that test root.
- `reads/*.npz` directly in `reads/`: first frozen read with z-scoring, invalid. Use `reads/own/` and `reads/v2/`.
- `e2e/cropopt_both_f0/f0_s47`: fold 0 seed 47, stopped after one epoch, no read.
- `source/run_e2e_gpu2_v2_20261107.sh`: never run, has lane GPU 1's quoting bug.
