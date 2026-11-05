# Why the score stopped falling: ceiling meta-analysis (2026-11-03)

Scope: out-of-fold (OOF) predictions of the shipped hybrid B (0.5 recipe A + 0.5 Phase-1 five), its members, the green
detector features, new train-row inference and the leaderboard result of submission 1. Score = mean of per-metric MAPE,
lower is better. Sources: [`source/`](source/), [`ceiling_meta.json`](ceiling_meta.json),
[`ceiling_meta_followup.json`](ceiling_meta_followup.json), [`stacker_deployable.json`](stacker_deployable.json),
[`stretch_nested.json`](stretch_nested.json), [`cdtail_images/`](cdtail_images/).

## Bottom line

The weak link is the bridge between image and regression target. Recipe A summarises the whole frame into a texture statistic (std of patch tokens). The labels are a statistic of individual cells in a region the technician could see. CD survives that mismatch; CV and HEX do not. A bigger or better-pretrained encoder, more seeds and more capacity over image tokens do not fix this: given more freedom the model memorises the training labels and validation gets worse.

Updated 2026-11-05. An earlier version concluded that image-level labels alone cannot teach the bridge. That no longer holds. A network over the cell graph of the most legible windows, built from detectors that saw no challenge labels and trained on image-level labels only, adds -0.056 (CI -0.074, -0.038) on top of B + hex6 + stretch, 5 of 5 folds, including the first cell-level CV gain (S14, S18). It works on geometry; image values at the nodes bring the memorisation back (+0.14 to +0.16). All of it is OOF, with configurations compared on the same folds.

| Metric | OOF B | Test (submission 1) | Log-space R2 | Prediction SD / label SD | Train-row error (folds 2, 3) | Validation error (folds 2, 3) |
|---|---:|---:|---:|---:|---:|---:|
| CD | 6.31 | 6.25 | 0.74 | 0.82 | 5.85, 5.81 | 5.90, 7.04 |
| CV | 10.20 | 10.14 | 0.46 | 0.63 | 9.66, 9.59 | 10.82, 10.14 |
| HEX | 10.12 | 10.51 | 0.36 | 0.57 | 9.76, 9.75 | 10.31, 10.27 |
| Mean | 8.88 | 8.97 | | | 8.43, 8.38 | 9.01, 9.15 |

Train and validation columns are the seed-46 recipe A member, epoch 20, one view, no calibration.

## Hypotheses tested

| # | Hypothesis | Test | Result | Verdict |
|---|---|---|---|---|
| H1 | Model variance limits us | Seed-to-seed variance; 1/S extrapolation | Seed variance is 0.7-1.5 % of single-member log MSE; 1 -> 2 seeds 8.975 -> 8.951; infinite seeds 8.927 (-0.024) | No. Small remainder only |
| H2 | Encoder strength limits us ("SSL weight class") | Error overlap between unrelated model families | A vs Phase-1 error correlation CD 0.91, CV 0.97, HEX 0.97; 95-98 % of B's error variance is shared; recipe A alone beats Phase-1 by only 0.05 | No |
| H3 | Fit: the model memorises and fails to generalise | Train-row inference with fold checkpoints | CV and HEX train error is only 5-11 % below validation; HEX train error 9.75 sits above the 8.6 sampling floor | Not at this learning rate. The fit level is set by the recipe, not by a capacity limit: see H11 |
| H4 | Domain or style bias | Year x style group offsets, 14 groups | 0.8-2.6 % of log MSE; oracle group offsets change each metric by -0.02 to +0.11 | No |
| H5 | Output calibration | Slope of log y on log B; LOFO stretch | Slopes 1.06 / 1.08 / 1.07. Stretch: A +0.004, Phase-1 -0.017, B -0.018 (CI -0.028, -0.007) | Yes, small and real. Averaging compresses the spread |
| H6 | A few images dominate through label-image disagreement | Lorenz share; visual check of the worst CD images | Top 1 % of CD errors (90 images, mean APE 66.9 %, 88 % overpredicted) = 10.6 % of CD error (Gaussian reference 3.6 %) = 0.22 pp of the final score | Yes for CD. Not fixable from the image (see below) |
| H7 | Per-image label noise (ROI sampling) | Fellow-eye residual correlation | Calibrated residual correlation CD 0.53, CV 0.29, HEX 0.09: HEX error is nearly independent per image, consistent with a sampling floor (~8.6 from the 25 annotated images) | Yes for HEX, floor below our error |
| H8 | The bridge reads global traits, not this eye's cells | Fellow-eye correlation of predictions vs labels | Predictions CD 0.80 / CV 0.78 / HEX 0.76; labels CD 0.82 / CV 0.43 / HEX 0.25. See note | **Yes for CV and HEX** |
| H9 | The residual is learnable from what we already have | LOFO gradient boosting on B's residual, nested shrink, no case-ID features | Predictions + image statistics -0.001 (nothing); adding detector features -0.038 to -0.042 (5/5 folds) | Only detector (cell) information helps |
| H10 | Better routing between existing models | Per-image oracle selection | A or P1 oracle 7.82, but y lies between A and P1 on only 7-12 % of rows and nothing available predicts which is right (H9) | Not reachable |
| H11 | Bridge capacity limits us | Fold 0, seed 46: encoder LR x5 (`enc_lr5x`); per-token residual MLP before std pooling (`token_mlp`) | `enc_lr5x` train 8.30 -> 7.26 (HEX 9.12 -> 7.86, below the 8.6 floor) but validation 9.11 -> 9.44 (+0.33, CI 0.21, 0.46). `token_mlp` train 8.32, validation 9.10 (-0.013, CI -0.051, 0.027), error correlation with recipe A 0.996 | **No.** More fitting memorises; it adds no signal that generalises |

Note on H8: label correlations are attenuated by label noise. With the HEX floor taken as noise (reliability ~0.54), the
true fellow-eye HEX correlation would be ~0.46, still far below the 0.76 of the predictions. The model's CV and HEX
outputs follow who or what the image is more than the cells of that eye.

## Side channels and physics probes (2026-11-04)

Prompted by the PI's objection that recent work was all incremental. Most of these failed; the ones still open are marked.

| # | Idea | Test | Result | Verdict |
|---|---|---|---|---|
| S1 | The case-ID order holds information the model cannot see | Pairs at ID distance d; `source/id_sequence_probe_20261104.py` | Old-format d = 1: label r CD 0.48, CV 0.41, HEX 0.35; residual r 0.38, 0.31, 0.22; both decay by d = 2-5. Knowing the neighbour's error: 9.04 -> 8.45 on 3777 old-format rows, 8.65 -> 8.40 on 4368 new-format rows. The largest effect ever measured here, but it needs the ID | Real, unusable as is |
| S2 | An image can find its partner | Raw descriptors, whitened SSL statistics, InfoNCE matcher trained on known pairs; `source/retrieval_matcher_20261104.py` | Matcher: new-format median rank 33-37 of 8953, top-1 11-13 %; old-format 262-394, top-1 3-6 %. kNN residual correction: -0.002 (k = 1), control -0.0004. On old-format rows -0.003 | **No.** Precision far too low to carry a residual |
| S3 | MHA header leaks acquisition data | 25 local MHA headers | Identical fields, spacing 0.7716049, no ID | No |
| S4 | Fellow-eye differences expose what the model misses | Within- vs between-pair correlation of B; `source/cotwin_probe_20261104.py` | New-format pairs: within-pair r CD 0.70, CV 0.26, HEX 0.23; between-pair 0.91, 0.57, 0.56. Predicted difference spread 0.87x, 0.29x, 0.24x the label's, slope 0.80-0.93. B's small differences are real but it rarely dares one | Quantifies H8 |
| S5 | Frozen features hold unused eye-specific signal | Ridge on pair differences of SSL statistics, LOFO | Correction -0.0002; held-out within-pair r 0.03-0.11 | No for frozen features. Trained-encoder run queued |
| S6 | Topology read from image ridges beats Voronoi | Seeded watershed from pre-only peaks; `source/abs_tess_20261104.py`, `abs_compare_20261104.py` | 25 images, marker hull: HEX r 0.75-0.81 vs Voronoi 0.39, oracle 0.84. 1000 frames, whole frame: HEX r 0.46, residual r 0.095 vs Voronoi on the same peaks 0.53, 0.125 and ten-detector hex6 0.55, 0.14. Partial r after hex6 about 0 | **No.** The 25-image result did not survive labels |
| S7 | Two sources need separate treatment | `source/source_format_probe_20261104.py`, `source_head_probe_20261104.py` | Classifier AUC 0.998; CD error old 7.11, new 5.50, CV and HEX equal. Per-source calibration no better than global; per-source head on frozen features 9.97 vs pooled 9.82 | No |
| S8 | The model reads cell size as physics requires | Zoom a trained member; `source/scale_response_20261104.py`. d log CD / d log s should be -2, CV and HEX 0 | Median slopes CD -1.42 (IQR -1.56, -1.18), CV +0.09, HEX -0.05. Zoom has never been in training | Partly. Zoom training queued |
| S9 | Zoom TTA helps | Same data, 500 rows | Learned slope, s = 0.9, 1, 1.1: mean +0.02 vs s = 1. Error r with s = 1: 0.92 (CD), 0.98 (CV, HEX) | No |
| S10 | An image's own response is a signal | Slope vs residual on 500 rows; `source/scale_gain_20261104.py` | 500 rows: r 0.19-0.24 with the member's CD residual, half-split CD -0.17 to -0.20. All folds, LOFO on top of hex6 + stretch (`source/zoom_feature_lofo_20261104.py`): CD only +0.009 (CI 0.004, 0.016), all metrics +0.019 | **No.** The small-sample gain was stretch seen again |

| S11 | The model reacts to things that cannot change the labels | Blur, gamma, noise, rotation, shift on 500 rows; `source/perturb_response_20261104.py` | Mean change in log prediction: blur sigma 1 CD +1.6 %, CV +6.8 %, HEX -5.8 %; sigma 2.5 CV +12 %, HEX -10 %; noise sigma 6 CD -6.8 % (SD 7.9 %), CV -7.2 %, HEX +3.7 %. Rotation and shift about 0 on average | **Yes: sharpness and noise are shortcuts.** Invariance training queued |
| S12 | Sensitivity is spendable uncertainty | `source/perturb_uncertainty_20261104.py` | |response to noise| vs |CD residual| r 0.42; CD error by quintile 3.9 to 9.0 %. Residual bias per quintile about 0; scale or A-P1 weight per quintile +0.05 to +0.17 on 250 rows | Real signal, nothing to spend it on |
| S13 | Exact duplicates leak or expose label errors | `source/duplicate_audit_20261104.py`, `swap_scan_20261104.py` | 14 groups, 28 rows, all new-format, 0.41 % of the error. 3 pairs same case with other spelling, same labels. 11 pairs with different labels, 10 of them across folds: images filed under a mistyped ID. Swap scan shows no excess over random pairs | Real data-quality fact, too small for the score |

| S14 | The image can find the technician's region | `source/legibility_roi_probe_20261105.py`, `legible_window_features_20261105.py`, `window_hex_ablation_20261105.py` | 25 sets: window with most pre-only peaks IoU 0.59 with box, 0.57 with hull (random 0.24, 0.11; centre 0.51, 0.38). All frames: point CV r 0.21 -> 0.44, CD 0.39 -> 0.57. HEX residual fit on top of B + hex6 + stretch: three windows -0.032 (CI -0.046, -0.017), whole frame -0.011, centre -0.006, hex6 alone 0, position alone +0.016 | **Yes.** First signal beyond hex6 since 10-28 |
| S15 | Frequency-domain lattice needs no points | `source/freq_lattice_20261105.py`, `freq_eval_20261105.py` | r up to 0.39 with HEX. HEX -0.003 (CI -0.015, 0.010), CD +0.011, CV +0.017 | No |
| S16 | Model predictions can pick the ROI | `source/window_roi_probe_20261105.py` | Box rank 0.68; best window IoU 0.14 vs random 0.23 | No, matching E9 |
| S17 | The labels are functions of the marked cells | `source/assumption_audit_20261105.py` | CD = 1e6 / integer mean area. Label / marker density 0.97-1.43, large where markers leave gaps. hex6 of markers within 1-2 % of the label on clean sets. CV of neighbour distance r 0.90, area CV r 0.88 | The 7.2 % CD "oracle" was an estimator limit |

| S18 | Relations between cells carry label signal that no summary captures | `source/cell_graph_prep_20261105.py`, `cell_graph_sweep_20261105.py`, `graph_ensemble_20261105.py`, `graph_leak_check_20261105.py`; runs in `graph_sweep/` | No context, on top of B + hex6 + stretch: 0 layers +0.014, 1 -0.013, 3 -0.034, 5 -0.043 (10 seeds), 9 -0.021; other detector seed -0.046; whole frame -0.028; ensemble of two detectors, CV and HEX: **-0.052 (CI -0.069, -0.035), OOF 8.755**. Giving the network B's predictions lowers the gain to -0.025. New-format rows -0.065, old -0.039 | **Yes.** First cell-level CV gain (-0.040) |

The earlier claim that folds are donor-grouped holds for new-format IDs only: old-format rows carry `slide_id` = ID, and adjacent old-format IDs share a fold at the random rate (0.187). Rows whose adjacent ID was in training are not better predicted (9.05 vs 8.99 when unseen), so this does not appear to inflate OOF.

## Label-image disagreement in the CD tail

Five worst-case images were inspected by eye, plus two correct low-CD controls (small sample, one reader):

- `2020-0651 OSCN` label 718, B 2699 and `2020-0644 OSCN` label 826, B 2991: small dense cells, consistent with B.
- `0933-24` label 733, B 2776: no readable cells.
- `0412-19` label 1220, B 1134: clearly large cells, predicted correctly. True low density is not what the model misses.
- Several worst cases are both eyes of neighbouring donors (2020-0644, 0646, 0651), which suggests a batch-level
  recording issue rather than biology.

## Why the earlier bridge attempts failed

All earlier cell-level bridges drew their cell geometry from the same thin source: 25 annotated images, or a generic
segmenter (Phase-1 SAM/CPSAM pseudo-masks, fNLA watershed). Self-training then re-taught what the teacher already knew
(consensus F1 0.666 -> 0.679), and a count-only density head failed (r 0.27). Detector quality stayed 2-3x short of the
E4 break-even (misses 26-29 % vs ~8 % for HEX; false positives ~19 % vs 2-11 % for CV). Point-based Voronoi CV is also a
biased proxy of device CV (0.49x), so CV needs cell boundaries, not only centres.

## Which experiments ever moved the score

| Idea class | Examples | Effect on the mean |
|---|---|---|
| Averaging partly independent errors | Hybrid B vs recipe A -0.074; two seeds -0.024; snapshot epochs 11-20 at member level -0.036 dev, -0.050 fold 4 | Paid, now nearly exhausted |
| One new signal from cells | hex6 from detector topology -0.031 | Paid |
| Output calibration | Stretch on B -0.018; after hex6 another -0.037 (total -0.068, CI -0.085, -0.052, fold 4 -0.063) | Paid, small |
| Reading the same image differently | Masked pooling +0.031 on framed; style specialist +0.068 on framed; box pooling +0.07 to +0.10; crops +0.002 | Zero or worse |
| Better detectors without new supervision | E1 CPSAM +0.0006, E2 -0.0048 (sanity failed), E7 threshold 0, E8 sharper targets +0.0001 | Zero |
| Point-based CV or CD | D1, E5, E6, p0, p9 (w = 0) | Zero |

Everything that paid either averaged existing information or added cell information. Nothing that re-read the same
texture paid.

## What follows

1. Post-processing is a part, not a plan. Stretch and hex6 can be added to any final model; they do not change the ranking
   on their own. The hex6 + stretch number was found while looking at all five folds, so it has no untouched
   confirmation fold.
2. The open question was a cell-level bridge built without human labels. Every line below failed its gate as of 2026-11-04; the cell graph of 2026-11-05 passed its own (S18):
   - **Bridge capacity**, closed (H11, one fold and one seed): more capacity does not generalise.
   - **Synthetic endothelium** (route B, E10), closed: a detector trained only on the v5 renderer scored all-25 miss 0.182, FP 0.542,
     jitter 0.309 against shipped pre-only 0.150, 0.165, 0.215. The renderer tuning target of F1 ~0.64 was a stale number; the
     comparable 12-px F1 on real val-5 is 0.857, so the final renderer was harder than real on misses, not matched.
   - **Detector without new human labels** (E11), closed: pre-only used no challenge labels and is the best detector so far. A noisy
     student (miss 0.196, FP 0.095, jitter 0.214) and SREP + synthetic (0.101, 0.264, 0.253) only move along the same F1 of ~0.86
     (pre-only 0.150, 0.165, 0.215). Jitter does not improve, though part of it may be annotator click noise that this
     score cannot separate.
   - **Label verifier** (route A), closed: CD + HEX cannot pick the ROI or trustworthy cells out of our detections. The
     expert box ranks like a random window (30-38th percentile) and own labels beat within-stratum nulls at 0.082 vs chance
     0.048 (green E9). Phase-1's version (fNLA watershed, CD + CV) had already reached only IoU ~0.2.

## Caveats

- OOF Phase-1 members used validation-selected checkpoints, so B's OOF is slightly optimistic. The test-minus-OOF gap sits
  entirely in HEX (+0.39), within the random-1000-image range 9.57-10.61. Whether this optimism causes it is unproven.
- The stacker's training rows have residuals from models that saw the held fold's labels (second-order leakage, not
  removed). The stretch and hex6 pipeline is properly nested.
- H11 is one fold, one seed and one setting per arm. It does not rule out a structured head that computes the label formula.
- Noise floors come from 25 annotated images, all 2023 framed style. No floor is measurable for CV.
- The train-fit table above shows folds 2 and 3. Folds 0, 1 and 4 agree (epoch 20 train mean 8.42, 8.48, 8.42; validation 9.11, 8.91, 9.05). Train error barely moves after epoch 10, where the learning rate reaches its floor. `enc_lr5x` confirmed the plateau comes from
  the schedule: at 5x the train error falls to 7.26 while validation rises. Calling this underfitting was too strong.
- SWA of epochs 11-20, four flips: member -0.033 (CI -0.049, -0.019; folds -0.073, -0.043, +0.015, +0.009, -0.075), but B with seed 46 swapped
  only -0.003 (CI -0.007, 0.000). Swapping seed 47 as well was not run. Closed: the ensemble already removes this variance.
