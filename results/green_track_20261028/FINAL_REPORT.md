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

## 6. Addendum 2026-10-30 — PI tasks G1/G2/G3 (hex6 adopted for submission 2)

PI decisions (2026-10-30): submission 1 = hybrid with the Phase-1 six; submission 2 = hybrid + hex6 correction;
GPU 2 and GPU 3 assigned to the green track until the deadline; external point data approved (Daniel et al.,
Sci Rep 2019; citation obligatory, see deploy/CITATIONS.md).

| task | what | result |
|---|---|---|
| G1a | GT-scored checkpoint selection (r of detector hex6 with GT HEX on 400 fold-train rows) | negative: r 0.38-0.56 is noise-limited and does not track held-out F1; 14-detector ensemble on B -0.0282 (unchanged); 2000-row re-test queued |
| G1 subset | LOFO-consistent selection among 13 detector families | negative: k = 11-13 chosen every fold, B -0.0272 |
| G1b | differentiable soft-hex6 surrogate for a weak HEX loss | failed its 20-image faithfulness test (r -0.15 vs discrete hex6); parked |
| G2 | external data survey | srep_topcon_sp3000 (385 specular images with cell-centre points, research use + citation) approved; rodrep confocal secondary; two others not usable |
| external round 1 | pre-train stride-2 head on 307 srep images (2000 updates, 1.30x rescale) then the 20-image fine-tune | no change: F1 0.609-0.671, coverage identical (hull 0.109 vs 0.110), 10-detector B -0.0286 |
| external transfer check | pre-trained head WITHOUT fine-tune, val-5 F1 | 0.54-0.59: external points transfer across device and scale |
| external round 2, pre-only member | nine + the pre-trained-only detector (never saw the 20 images) | **GATE PASS on B: -0.0934 pp HEX = -0.0311 pp mean, HEX 5/5 folds, 3/5 folds <= -0.03 (per fold -0.047 / -0.031 / -0.042 / -0.027 / -0.009); A -0.0515 (4/5)** |
| external round 2, other variants | spacing-matched scale, 6000-update pre-training, joint training | spacing folds 4/0/1 F1 .652/.618/.631 unchanged; long F1 .657/.651/.637, transfer .54-.56; joint pending; blends pending |
| G3 | nine submission-2 detectors from the full-data (fold-99) encoder, full-fit a/b/w on B, Hex6Corrector, tests | delivered and frozen: a -0.223, b 0.584, w 0.26 (8921 rows; LOFO -0.0274 mean on B); unit test and container regression test (25-image fixture) pass; 0.68-0.75 s/image |

Multiplicity caveat (to be weighed by the PI): the pre-only member is one of five pre-registered round-2
variants (protocol_round2.json and its amendment of 17:35 UTC, written before any of them ran) and its gate pass
is marginal on the mean (-0.0311 vs the 0.03 bar). A confirmation with two more pre-only seeds and a
12-detector blend is queued; the parent's fixed decision rule: 3-pre-only ensemble on B <= -0.035 with >= 3
folds <= -0.03 and HEX 5/5 -> nine + three pre-only; -0.030..-0.035 -> nine + the best single pre-only;
> -0.030 -> the nine stay frozen. Mechanism: independence — every other detector was fine-tuned on the same
20 images; the external-only detector's hex6 errors are less shared, which is why the PI pushed for external
data. The deployed pre-only versions are pre-trained from the fold-99 encoder with threshold 0.35, fixed without
the 5 validation images (the same threshold every detector used in the OOF evaluation).

Still open after this: the CV token-field question (Voronoi-free statistics of the recipe A token field as CV
features; token_field_cv/protocol.json, queued), then nothing else detector-side by agreement.

## 7. Addendum 2026-10-31 — confirmation and submission-2 configuration

Confirmation (pre-only seeds 47/48 added): nine + three pre-only on B = -0.0978 pp HEX = -0.0326 pp mean, HEX
5/5, 4/5 folds <= -0.03 (per fold -0.0489 / -0.0374 / -0.0404 / -0.0306 / -0.0058); A -0.0533. Singles as
tenth member on B: s46 -0.0311 (3/5), s48 -0.0305, s47 -0.0303. Parent's fixed rule (middle band) ->
**submission 2 = nine + pre-only s46**; activated parameters a -0.2084, b 0.5989, w 0.28 (8925 rows); the
deployed member is pre-trained from the fold-99 encoder (2000 updates, threshold 0.35 fixed without the 5
validation images). Version-2 unit test (0.80 s/image, ten detectors; feature r 0.74 with expert hex6, 0.78
with GT HEX) and container regression test (zero drift, 0.88 s/image) pass; fixture regression_fixture_25_v2.npz.

Closed in the same round: G1a (2000-row re-test still flat), joint training (-0.0274), spacing-matched
(-0.0294) and 6000-update (-0.0299) fine-tuned variants — the 20-image fine-tune removes the external benefit;
token-field CV statistics (primary weight 0) — CV closed. Multiplicity caveat unchanged (section 6).

## 8. Addendum 2026-10-31 (later) — deployment checks and the frozen-encoder alternative

- Encoders: the ten deployed detectors each carry their own fine-tuned encoder (weight deviations ~1e-3 to
  4e-3 from the fold-99 encoder and from each other; none equals a full-data recipe A member), so no shared-
  encoder path exists for them. Container cost ~3.4 GB and ten extra ViT-B passes (0.80-0.88 s/image on an H200).
- Strata (paired donor-cluster bootstrap, active configuration vs B): all -0.0311 [-0.044, -0.019];
  black_frac > 0.01 -0.0212 [-0.041, -0.001]; black_frac <= 0.01 -0.0403; years 2023-2024 -0.0322
  [-0.066, -0.000]; 2018-2022 -0.0309. Accepted by the parent: expect about -0.02 pp on black-frame images.
- Fold-4 transfer: the deployed parameters applied unchanged to the shipped hybrid (three-seed + six) give
  -0.0139 pp (HEX -0.042) vs -0.0143 on B; the fold-4 LOFO fit reproduces -0.0086. Parameters transfer.
- Frozen-encoder alternative (heads on the hybrid's own s46 tokens, zero extra passes): -0.0112 pp on B versus
  -0.0311 for the active ten; mixing frozen heads into the ten dilutes it (-0.0267). Decision left to the PI/parent.
- GPUs 2 and 3 handed back to the parent at 16:42 UTC; the green track has no GPU work left.

## 9. Addendum 2026-11-01 — T4-container numerics check (verdict: shipped hex6 stays)

The submission container runs on a T4 (no flash attention, no native bf16), so the fold-clean OOF hex6 features
of the active families were re-extracted under the container numerics (fp16 autocast, SDPA restricted to
efficient/math, heads float32, peak finding unchanged) on lane01 (GPUs 0-1, 13:15-14:49 UTC) and the LOFO
blend on base B was rerun (CPU, 14:49-15:23 UTC). Seven of the ten active families have fold-clean checkpoints;
s8_s47, s8_s48 and s8multi_s47 were never saved and, by the parent's decision, were not retrained — their bf16
features are kept in the ten-family numbers, which are context only.

- Decisive, seven families on the same checkpoints: bf16 -0.0291 pp mean (HEX -0.0874) vs fp16 -0.0281 pp
  (HEX -0.0844); per fold bf16 -0.0440 / -0.0320 / -0.0398 / -0.0244 / -0.0055, fp16 -0.0412 / -0.0265 /
  -0.0414 / -0.0258 / -0.0058; LOFO w stays in 0.24-0.32. Difference +0.0010 pp, within the 0.005 limit.
- Deployed a = -0.2084, b = 0.5989, w = 0.28 applied unchanged to fp16 features (ten, mixed), all 8954 fold-clean
  rows: -0.0326 pp mean (HEX -0.0979), per fold -0.0519 / -0.0265 / -0.0429 / -0.0293 / -0.0128, 5/5 folds;
  on bf16 ten -0.0331. Not worse than the -0.025 limit.
- Context, flagged: ten-family LOFO bf16 -0.0311 vs fp16-where-available -0.0297.
- Feature stability (log hex6): per detector r 0.90-0.92 (pre-only sg2pre_s46 0.96), mean |delta| 0.04-0.05,
  96-99% of rows change; ensemble mean r 0.980 (seven) / 0.989 (ten), mean |delta| 0.019 / 0.013, max 0.52 on one
  row. fp16 moves individual detectors about as much as a seed change, the ensemble mean and the gain survive.
- Verdict accepted by the parent 15:35 UTC: nine + pre-only s46 with the deployed a/b/w stays for submission 2.
  The bf16-only code path in `deploy/hex6_correction.py` is handled outside this module by the container's
  `submission2_test_20261101/source/hex6_t4_20261101.py` subclass (fp16 + efficient/math SDPA), with 25-image fixture parity and a 60-image
  fold-4 integration run (CD/CV unchanged, hex6 applied 60/60, deployed-vs-OOF feature r 0.967) verified by the
  parent. `deploy/` is frozen and not to be edited. Files: `t4_numerics/{result.json, verdict.md, protocol.json}`.
- Green track status: closed; daily summary only, no further GPU work planned.

## 10. Addendum 2026-11-02 — green day 2 (all four GPUs, PI directive of 2026-11-01)

Full ranked summary with gates and files: `experiments/green_day2_20261101/SUMMARY.md`. In one table (mean pp on B,
fold-clean, 20 leaked rows excluded; adoption bar for submission 2 is 0.02 pp, so everything here is evidence-only):

| rank | finding | effect | gate |
|---|---|---|---|
| 1 | coverage-conditional hex6 weight (w per framed x detection-count tercile, 0.14-0.50) | -0.0374 vs -0.0311 (gain -0.0062, 5/5 folds) | passed (>= 0.005), not adopted (< 0.02) |
| 2 | distilled hex6 from the regressor cannot replace the detectors (r 0.855, blend w = 0) | +0.0001 | fail; detector passes stay necessary |
| 3 | detector CD capped by the expert-point oracle (7.2% vs device CD; B 7.7% on the same 25); all CD estimators w = 0 in every stratum | 0.0000 | stop rule fired; CD closed |
| 4 | more independent full-frame members saturate (three -0.0337, five -0.0334) | -0.0023..-0.0026 | fail |
| 5 | frame-augmented pre-only detector as member | -0.0293 swap / -0.0315 add | fail |
| 6 | hex6-auxiliary multi-task fine-tune of recipe A (lambda 1 and 3) | -0.0019 / +0.0080 at B level, aux = baseline | fail |
| 7 | tissue-interior / window / tissue-filtered hex6 variants | no change or framed-only gain with unframed loss | fail |

GPU use 20:10-02:10 UTC on all four GPUs through the parent's lane runners; GPUs returned to keepalive at 02:10 UTC.

## 11. Addendum 2026-11-03 — green day 3 (per-cell-geometry bottleneck; all four GPUs)

Full summary: `experiments/green_day3_20261103/SUMMARY.md`. Submission 1 scored 8.9675 on Phase II (CD 6.250 / CV 10.143 /
HEX 10.510; parent), within sampling noise of the OOF, so the OOF harness is a validated test proxy. Day 3 ran eight
pre-registered experiments; none passes a gate. The decisive one is E4 (error decomposition on the 25 expert sets):
HEX loses its oracle edge at ~8 % misses / 0.17 x spacing jitter, CV at ~2 % split-type FPs, CD has no headroom; the real
detectors sit at 17-29 % misses and 0.20-0.25 jitter, CPSAM at 51 % misses. Robust CV estimators (E5), learned per-cell
heads (E2/E6), a lower threshold (E7), sharper detector targets (E8), CPSAM cells (E1) and window-restricted pooling (E3)
all fail on the 8954. The shipped hex6 (-0.0311) stays the only green contribution; its coverage-conditional weight
(-0.0374) remains the documented, evidence-only refinement. A feature export for the parent's post-processing search
is at `experiments/submission2_search_20261103/green_features.npz`.

**Addendum 2026-11-04 — label-free routes.** With human annotation ruled out by the PI, three label-free attempts at a
cell-level bridge were run and all fail their pre-registered gates: the label verifier (E9, route A) ranks expert windows
at the 30-38th percentile of random windows; the synthetic-only detector (E10, route B, renderer v1-v5) reaches miss 0.182 /
FP 0.542 / jitter 0.309 on the 25 real images; the noisy student of the shipped detector (E11a: miss 0.196 / FP 0.095 /
jitter 0.214) and the SREP + synthetic mixed model (E11b: 0.101 / 0.264 / 0.253) move in opposite directions from the
shipped 0.150 / 0.165 / 0.215 without reaching recall >= 92 % at < 4 px. Green day 3 is closed with the shipped hex6 blend
unchanged. Operational note: the first E11 run OOM-killed the research pod (an npz re-read bug in `load_pseudo()`, fixed;
see `e11_nolabel/verdict.md`); the rerun on the replacement pod used 6.6 GB.
