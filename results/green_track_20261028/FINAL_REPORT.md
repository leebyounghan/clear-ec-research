# Green track on GPU 3 — final report (2026-10-28 15:40 UTC to 2026-10-29 19:00 UTC)

Scope: the 25 expert-annotated images (cell-centre points and ROI boxes; 20 train, 5 validation, all inside
fold 4 as recorded in experiments/green_point_probe_20261023/data_binding.json) as a source of weak supervision
or features for the CLEAR-EC Phase 2 pipeline. Every number below is a five-fold out-of-fold (OOF) delta in
percentage points of the mean MAPE score (lower is better) on the 8934 rows whose point labels no head ever
saw, with leave-one-fold-out (LOFO) fitting of any calibration or blend parameter, against two bases:

- Base A: recipe A seed-46 single model, four-flip TTA, LOFO scalar calibration: 8.9757 on the 8934 rows
  (8.985 on all 8954; CD 6.396 / CV 10.278 / HEX 10.253).
- Base B: submission-1 hybrid, 0.5 recipe A (two seeds) + 0.5 Phase-1 five (calibrated): 8.8682 on the
  8934 rows (CD 6.312 / CV 10.204 / HEX 10.116 on all rows).

Gate: >= 0.03 pp mean improvement overall and on >= 2 folds, no target worse by > 0.05 pp. Parent's decision
rule for the submission base: >= 0.03 pp on B with HEX improving on >= 4 folds.

## 1. All ideas and their OOF deltas

Negative deltas are improvements. "n/a" means the experiment was evaluated on base A only (its verdict did not
depend on the base) or produced no OOF number by design (pilot or oracle). Folders are under
experiments/green_track_20261028/; each has protocol.json (written before running), result.json and verdict.md.

| # | Idea (folder) | What was tested | Delta on A (pp mean) | Delta on B (pp mean) | Verdict |
|---|---|---|---|---|---|
| 1 | CD estimators from existing point maps (p0_cd_estimators) | 28 grid-level and sub-grid spacing estimators, LOFO log-blend into CD | best +0.002 (sub-grid spacing: CD MAPE 8.9%, error corr 0.72, w 0-0.04) | n/a | negative; grid features carry no CD signal |
| 2 | Voronoi oracle on expert points (p1_voronoi_oracle) | CV/HEX from perfect points, 25 images; degradation with misses/FPs | oracle only: CV 8.8%, HEX 7.8% LOO-scaled; 15% miss + 15% FP collapses CV r to 0.08 | n/a | prize exists only with near-perfect points |
| 3 | Voronoi features from real detections (p2_voronoi_features) | hex6, area CV, local CD from the stride-8 specialist, LOFO per target | HEX -0.015 (hex6); CV 0.000; CD +0.002 | n/a | first HEX signal; CV/CD nothing |
| 4 | hex6 definition variants (p2b_hex6_variants) | six definitions (completeness filters, Gabriel graph, weighting) | best -0.018 (all bounded cells) | n/a | variant F chosen a priori for confirmation |
| 5 | Quality-conditioned calibration scalars (p3_quality_calibration) | tercile MAPE-optimal scalars keyed on detector-derived quality | +0.001 to +0.001 | n/a | negative |
| 6 | Self-training round 1 (st1_fold4) | single confident teacher, stride-8 student, fold-4 pilot | val-5 F1 0.642 -> 0.666; downstream r unchanged | n/a | not extended |
| 7 | Stride-2 point head (sg2_specialist) | pixel-shuffle head, 5 folds, confirms hex6 on an independent detector | HEX -0.021 (hex6_all, 5/5 folds); CD 0.000 | -0.009 | confirmation of the HEX lead |
| 8 | Claude audit of residual extremes (vlm_audit) | contact sheets of worst/best HEX residuals with detections | hypotheses only | n/a | detector = ROI proposal; worst HEX cases are pathological corneas where hex6 corrects |
| 9 | Cluster-crop re-prediction (roi_crop) | recipe A on 512x512 / 736x992 crops around the detector cluster | +0.002 / +0.003 | n/a | negative (crop errors 0.81-0.97 correlated) |
| 10 | Region-local token pooling (local_pool) | ridge head on probe-weighted / box tokens vs global control | global control -0.019; soft -0.016; box -0.010 | n/a | negative for the region; control gain is head diversity |
| 11 | Fitted-hexagon IoU and sides features (p4_hexshape) | shapely hexagon IoU, sides histogram, edge CV | primary hexiou_mean -0.003; exploratory mean|sides-6| -0.023 | n/a | shape detail is noise; sides deviation ~ hex6 |
| 12 | Two-detector hex6 average (p5_hex6_combo) | geometric mean of hex6 over stride-8 + stride-2 (parameter-free) | -0.0305 (gate pass, 2 folds <= -0.03) | -0.015 | absorbed by the hybrid |
| 13 | Stride-2 TTA and seeds 47/48 (sg2_tta_seeds) | four-flip heatmap TTA; head seeds | TTA -0.019 vs -0.021 single; seeds -0.017 / -0.023 each | n/a | TTA does nothing; seeds are independent looks |
| 14 | Multi-detector hex6 ensemble (p6_hex6_ensemble) | mean log hex6 over 5 / 9 / 13 fold-clean detectors | -0.041 / -0.048 / -0.047 (4/5 folds <= -0.03) | -0.023 / -0.028 / -0.0286 (5/5 improving, 2/5 <= -0.03) | saturates at 9; below the 0.03 bar on B; OUT of submission 1 |
| 15 | Extra detector seeds and architectures (detector_seeds_overnight, detector_diversity_overnight) | stride-8 s47/s48, stride-2 s49-s52, multi-layer stride-8 heads | each -0.013 to -0.017 alone; no ensemble gain beyond 9 | (see 14) | diversity across stride helps more than seeds |
| 16 | Token ridge + hex6 three-way blend (p7_token_hex6_ridge) | base + frozen-token ridge HEX + hex6 | -0.034 (hex6 alone -0.0305) | -0.015 (= hex6 alone) | redundant |
| 17 | Density/count head, weak CD integral (density_head, backlog item 1) | strong points + weak frame-integral labels, fold-4 pilot | r 0.27 with GT CD, 52% MAPE, no localisation | n/a | negative; closed |
| 18 | Consensus-teacher self-training (st2_consensus) | pseudo-points from >= 3 of 4 detectors, stride-2 student, fold-4 pilot | val-5 F1 0.666 -> 0.679 (rule 0.70) | n/a | not extended |
| 19 | Uncertainty-conditioned CV calibration / hybrid weight (p8_uncertainty_cv) | four disagreement proxies x two variants | -0.0001 to +0.002 | +0.001 to +0.004 | negative |
| 20 | Consensus-point Voronoi CV (p9_consensus_cv) | area CV from 13-detector consensus, k = 5/7/9 | 0.000 (w = 0; CV MAPE 14.3%, r_err 0.77) | 0.000 | negative; CV from points closed |
| 21 | ROI-box token re-weighting (roi_reweight) | not run | - | - | dropped: Phase 1 ROI recovery IoU 0.2, true ROI helps CV 0.35 only |

Side result worth keeping: the earlier ridge stacking's +0.07..+0.10 pp "harm" was mostly its L2 log-intercept
calibration (+0.059 pp by itself against the MAPE-optimal scalar); those point-map features were "no signal",
not harmful.

## 2. The hex6 detector-count curve

Feature: six-neighbour fraction of the bounded Voronoi cells of detected cell centres (threshold 0.35,
centroid-refined peaks), averaged in log space over N fold-clean detectors; per held fold, log HEX = a + b log
hex6 and one blend weight w are fitted on the other four folds and applied as exp((1-w) log base + w log est).

| detectors | composition | A: dHEX / dMEAN (folds mean <= -0.03) | B: dHEX / dMEAN (folds mean <= -0.03; HEX improving) | w on B |
|---|---|---|---|---|
| 1 | stride-8 s46 or stride-2 s46 | -0.054 / -0.018 (0) ; -0.063 / -0.021 (1) | -0.023 / -0.008 (0; 5/5) ; -0.026 / -0.009 (0; 5/5) | 0.12-0.16 |
| 2 | stride-8 + stride-2 | -0.091 / -0.0305 (2) | -0.045 / -0.015 (0; 5/5) | 0.16-0.20 |
| 5 | + stride-2 s47, s48, TTA | -0.124 / -0.041 (4) | -0.070 / -0.023 (1; 5/5) | ~0.22 |
| 9 | + stride-8 s47, s48; stride-2 s49, s50 | -0.145 / -0.048 (4) | -0.085 / -0.028 (2; 5/5) | 0.24-0.28 |
| 13 | + multi-layer stride-8 s46, s47; stride-2 s51, s52 | -0.142 / -0.047 (4) | -0.086 / -0.0286 (2; 5/5) | 0.24-0.28 |

Per-fold mean deltas on B with 13 detectors: -0.041 / -0.026 / -0.038 / -0.026 / -0.013. The two detector
families' log hex6 correlate 0.68, which is why averaging helps; four-flip TTA of one detector does not
(-0.057 vs -0.063 pp HEX). The curve is flat from 9 detectors: the ceiling of this feature on the submission
base is about 0.03 pp, at the bar, not above it.

## 3. What deploying hex6 would require

- Inference: N extra fine-tuned CellposeDINO ViT-B forward passes per image (one per detector, ~0.14 s each
  on an H200 at 992x1312), peak finding, Delaunay/Voronoi on CPU (milliseconds), the calibrated blend. With
  N = 9 that is roughly 1.3 s per image on top of the regressor.
- Models: each detector is a full fine-tuned copy of a recipe A encoder (340 MB) plus a linear (stride-8) or
  Linear-768->16 + pixel-shuffle (stride-2) head, trained for 1000 updates on the 20 train-green images. For
  the submission these would have to be re-trained from the full-data recipe A encoder with the same recipe
  (6-8 min each on one GPU) because the fold-clean copies must not be used on test data.
- Parameters: per detector nothing beyond the fixed threshold 0.35; for the blend, a, b (log-linear
  calibration of the averaged log hex6 to log HEX) and w, fitted on all five OOF folds of the chosen base
  (full-fit values on B: w ~0.26; a, b per experiments/green_track_20261028/p6_hex6_ensemble/result_everything.json).
- Rows without a hex6 estimate (fewer than 20 peaks, ~1% of images) fall back to the base prediction.
- Expected effect on B: -0.086 pp HEX, -0.029 pp mean, consistent across folds but below the 0.03 bar;
  no effect on CD or CV. Decision (parent, 2026-10-29): not included in submission 1.

## 4. Why the track could not move CD or CV

1. Twenty annotated images bound detection quality. Every detector, whatever its stride, layer set, seed,
   TTA or self-training variant, plateaus at held-out point F1 0.62-0.69 on the five validation images
   (recall 0.67-0.81, precision 0.56-0.64). Self-training with a single teacher (+0.024 F1) or with a
   four-detector consensus (+0.012 F1) does not escape this; the pseudo-labels only re-teach what the
   detectors already agree on. The detectors also learn the annotators' ROI selection: they fire on one
   compact cluster of 100-300 cells in the best-focus region, not across the frame, so frame-level counts are
   uninformative for CD (best |r| 0.30 with the grid features; 0.73 with sub-grid spacing, but 0.72
   correlated with the regressor's own errors).
2. Misses merge cells, so area statistics are corrupted at any reachable detection quality. With the expert
   points, Voronoi area CV reaches 8.8% MAPE and would beat the regressor (10.3%); with 15% random misses and
   15% false positives its correlation with GT CV collapses to 0.08. Consensus across 13 detectors removes
   false positives but not misses (consensus CV MAPE 14.3%, blend weight 0). Only a neighbour-count statistic
   (hex6) survives, because a merged cell changes counts by one where it changes areas by a factor of two.
3. The regressor's CV errors are not predictable from disagreement. Neither the spread of its four TTA views,
   nor the disagreement between seeds, nor between the two halves of the hybrid, nor between the 13 detectors'
   hex6 conditions a useful calibration or hybrid weight (all within +-0.004 pp). Whatever limits CV at ~10%
   on the submission base is not an uncertainty that these signals can see; the intra-slide GT spread (5.3%
   relative) says it is not pure label noise either.

## 5. Status

GPU 3: queue empty since 2026-10-29 12:52 UTC; the queue runner (pid 41745, spec gpu3_jobs.json) holds
gpu3.lock and runs the keepalive. Awaiting the PI's word via the parent session. Ledger:
experiments/green_track_20261028/ledger.json; local copies of every small JSON, protocol and verdict in
results/green_track_20261028/ of the clear-ec-research repository.
