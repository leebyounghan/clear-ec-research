# Lane GPU 1: SSL with strong, domain-specific views (plan row G2)

Gates in `protocol.json`. Sources in `source/`. Reads in `reads/`. Times Eastern.

## Frozen read of existing encoders, 2026-11-06

Ridge on the pooled tokens, no fine-tuning. Fold 4, 1790 rows.

| Encoder | std | mean | End to end, mean pooling, one seed |
|---|---:|---:|---:|
| External CPDINO | 9.747 | 9.830 | 9.147 |
| SSL 5 epochs | 9.542 | 9.504 | 9.133 |
| SSL 10 epochs | 9.675 | 9.616 | 9.111 |
| SSL 20 epochs | 9.675 | 9.630 | 9.120 |
| Recipe A tuned, epoch 20 | 9.079 | 9.198 | |

The frozen spread is 0.33, the end-to-end spread 0.035, and SSL 10 and 20 change sign. Fine-tuning removes most of a frozen difference, so a frozen delta ranks starting points only.

## Stage 1, fold 4, frozen read against SSL 5, std pool

All arms: 5 epochs, LR 1e-5, seed 1003, crop 544 px, about 41 min including features.

| Arm | Delta | CI | CD | CV | HEX |
|---|---:|---|---:|---:|---:|
| Crop, SSL pools by mean | -0.094 | -0.191, +0.009 | -0.261 | +0.080 | -0.100 |
| Crop, SSL pools by std | -0.139 | -0.235, -0.044 | -0.336 | +0.034 | -0.115 |
| Crop + broad optics, mean | -0.159 | -0.257, -0.058 | -0.263 | -0.036 | -0.176 |
| Crop + broad optics, std | -0.157 | -0.261, -0.052 | -0.389 | -0.005 | -0.077 |
| Crop + broad optics, mean and std | -0.166 | -0.262, -0.068 | -0.426 | +0.010 | -0.081 |

One fold and no seed check yet, so nothing is decided. Most of the gain is on CD, the metric that fine-tuning moves most.

## Stage 2, fold 0, frozen read against SSL 5

Crop + broad optics, SSL pools by mean and std. Chosen on fold 4 alone, read on fold 0 once.

| Read | Delta | CI | CD | CV | HEX |
|---|---:|---|---:|---:|---:|
| Fold 0, std pool | -0.198 | -0.300, -0.096 | -0.466 | -0.025 | -0.104 |
| Fold 4, std pool | -0.166 | -0.262, -0.068 | -0.426 | +0.010 | -0.081 |
| Two folds pooled, std pool (primary) | -0.182 | -0.252, -0.112 | -0.446 | -0.008 | -0.093 |
| Two folds pooled, mean pool | -0.196 | -0.264, -0.126 | -0.381 | -0.093 | -0.114 |
| Two folds pooled, mean and std | -0.148 | -0.213, -0.083 | -0.324 | -0.023 | -0.096 |

The screen for an end-to-end run passes: <= -0.03, CI below 0, both folds negative. This is not a gain on the score. On fold 4 the frozen spread between SSL durations did not survive fine-tuning, and CD, where most of this delta sits, is the metric fine-tuning moves most. The seed check is still to run.

## End-to-end runs from this encoder

Recipe A, std pooling, 20 epochs, schedule 10, one GPU. Pass as in the plan: mean of the last five epochs, two folds, paired with the same-seed recipe A.

| Fold | Seed | Where | Root | Started (EST) |
|---|---:|---|---|---|
| 4 | 46 | GPU 1 | `e2e/cropopt_both_f4/f4_s46` | 11-06 23:25 |
| 0 | 46 | GPU 3 | `e2e/cropopt_both_f0/f0_s46` | 11-06 23:13 |
| 4 | 47 | GPU 2 | `lane_gpu2_20261106/e2e/cropopt_both_f4/f4_s47` | 11-06 22:57 |
| 0 | 47 | GPU 2, queued | `lane_gpu2_20261106/e2e/cropopt_both_f0/f0_s47` | about 08:00 |

## Incidents

- 11-06 18:56 to 22:35 EST: GPU 1 ran keepalive only, because the session's waiter ended and nothing was queued. Queue depth is now past 12:00 EST.
- 11-06 23:21 EST: `run_e2e_v2_20261107.sh` failed twice in 0 s. An apostrophe inside `${VAR:?message}` opened a quote that swallowed the setup lines; `bash -n` passes it. `run_e2e_v3_20261107.sh` is the fix. Fold 1 SSL was stopped twice in those minutes and requeued behind the end-to-end run. About 3 minutes lost.
- `ssl/cropopt_both_f0/regression_input` was first built from epoch 1 of 5 by another lane's start. Rebuilt at 04:13 UTC and checked tensor-equal to `run/epoch05.pt`. Launchers from v2 on rebuild it unless it is newer than the SSL.
- Left in place: `ssl/t_crop_smoke`, `ssl/t_opt_smoke`, empty `ssl/cropopt_both_s1004_f4`, `e2e/cropopt_both_f4/f4_s47` (wrapper test, failed exit.json from another lane), `e2e/cropopt_both_f0/INVALID_partial_ssl_f0_s47_20261107`. Deleting them is the PI's call.

## End-to-end verdict, 2026-11-07

Crop + broad optics encoder, recipe A, seed 46, mean of epochs 16-20, against the same-seed recipe A.

| Read | Delta | CI | CD | CV | HEX |
|---|---:|---|---:|---:|---:|
| Fold 0 | +0.034 | | +0.069 | +0.001 | +0.030 |
| Fold 4 | +0.004 | | +0.022 | +0.016 | -0.028 |
| Two folds | +0.019 | -0.020, +0.058 | +0.046 | +0.009 | +0.001 |
| Added to the best pipeline, w 0.125 | -0.004 | -0.010, +0.002 | -0.002 | -0.003 | -0.006 |
| Added to the best pipeline, w 0.25 | -0.003 | -0.015, +0.008 | +0.001 | -0.002 | -0.008 |
| Fold 4 seed 47, alone | +0.003 | -0.046, +0.055 | +0.047 | +0.029 | -0.066 |

Fails both pass rules. Promotion does not start. A frozen delta of -0.182 left nothing end to end, so belief 1 now has a direct test: a much stronger SSL by the frozen read does not change the final model. Fine-tuning lands on the recipe A solution (residual r 0.98-0.99 on fold 0).

The frozen ridge prediction itself as a member, folds 0 and 4: w 0.125 +0.001 (CI -0.011, +0.013), w 0.25 +0.022 (CI -0.001, +0.045). No gain either. Its residuals correlate 0.88, 0.95, 0.97 with recipe A's on CD, CV, HEX; this high value is partly arithmetic, because residuals share the label.

## Node tokens from the strong-view encoder, 2026-11-07

Belief 14: image values at graph nodes memorise. Tokens at nodes of the pre-only s48 graphs, against the same graph network with geometry only (d 64, 5 layers, 3 seeds, five folds, CV and HEX only). Gate: <= -0.01, CI below 0, 4 of 5 folds.

| Arm | Delta | CI | CV | HEX | Folds negative |
|---|---:|---|---:|---:|---:|
| Strong-view tokens, 8 dims | +0.069 | +0.051, +0.087 | +0.060 | +0.146 | 0 |
| SSL 5 tokens, 8 dims (control) | +0.079 | +0.059, +0.099 | +0.074 | +0.163 | 0 |
| Strong-view tokens, 32 dims | +0.138 | +0.109, +0.166 | +0.183 | +0.232 | 0 |
| Strong-view tokens, 8 dims, dropout 0.3, weight decay 0.1 | +0.070 | +0.052, +0.090 | +0.066 | +0.145 | 0 |

All fail. The control reproduces the earlier harm, more dimensions harm more, and stronger regularisation changes nothing. The token source moves the result by 0.010. Belief 14 stands for these tokens and this regulariser.

## State of plan row G2

Closed. Starting point: no gain end to end. As a frozen member: no gain. As node values: harm.

## Plan idea I7, donor memory from images, 2026-11-07

Encoder features whitened by the within-pair covariance of half of source B's fellow-eye pairs (linear only; no nonlinear pair network was trained). Among all 8954 frames, strong-view features: B held-out pairs top-1 17.4 %, median rank 31; A serial neighbours top-1 6.2 %, median rank 398. The draft's kill line is 40 % on A. Recipe A tuned features: 14.4 % and 4.1 %.

Direct read, correcting the best pipeline by neighbours' OOF residuals, K and shrink nested, five folds: strong-view +0.005 (CI +0.001, +0.010), tuned -0.002 (CI -0.007, +0.004). Both fail the correction gate, and this read assumes a test frame has its partner in training, which is unverified. `reads/donor_link.json`, `reads/donor_memory.json`.

## After G2

GPU 1 runs `dinov2_b14` fold 0 seed 46 for lane GPU 3, at lane GPU 3's root, from 2026-11-07 09:10. The read and gates are that lane's. `cropopt_both_seed1004_f4` was stopped 9 min in as low value: the seed check only mattered for calling an arm better on the frozen read, and the end-to-end verdict replaced that.

`dinov2_b14` fold 0 seed 46 ended 2026-11-07 10:57:39 EST, rc 0, 6548 s, 20 epochs, no NaN. Lane GPU 3 was told and reads it.

`cpsam_c512` fold 4 seed 46 runs on GPU 1 for lane GPU 3 from 2026-11-07 10:57 EST, at lane GPU 3's root, handed over by that lane, about 22 h by its estimate. The read and gates are that lane's. If GPU 1 is moved, lane GPU 3 takes fold 4 back.
