# Cell-route lane (device outline rule on detected cells), 2026-11-09

Goal: make the Konan centre-method rule (weighted bisectors, k=2) work on our detected cells, so that the CV and HEX lanes get a new input.
Folder: `experiments/lane_cellroute_20261109/`. `RESULTS.tsv` has one line per try. Code is in `source/`, features in `feats/`, predictions and gate JSON in `out/`.
CPU only, at most 4 workers. About 10 CPU-hours used.

## Conclusion (14:00 EST 11-09)
- **HEX: one small kept gain.** Konan k0/k2 statistics from a union point set (s46/s47/s48 pre-only TTA peaks >= 0.35 plus Cellpose-SAM mask centroids, cross-source vote, >= 1 source).
  Applied as a nested ridge on the HEX lane best, it gives **HEX 9.7315 vs 9.7380 (-0.0065)**.
  - Folds: -0.020 / -0.008 / -0.008 / +0.007 / -0.004. Folds 0+4: -0.012.
  - The point set was pre-registered from the 25-frame gate.
  - Not robust across versions: other versions built on similar features read between -0.0015 and +0.010.
- **CV: closed.** Three clearly different versions are all worse on the CV lane best, by +0.015 to +0.029:
  - cleaned points;
  - union points;
  - operator-like cell selection with outlier trim.
  Trimming does lift raw r with the CV label from 0.2-0.3 to about 0.50, but r with the stack residual stays about +0.03. The stack already holds that information.
- **Why detected points stop short of marker quality.** Two unrelated machine point sets, detector peaks and mask centroids, agree to 0.088 of spacing. Each differs from the expert clicks by about 0.185.
  - So most of that difference is click noise, and the device label is computed from those noisy clicks.
  - Per-frame click scatter correlates with the label on the 25 frames (CV +0.70, HEX -0.72).
  - Emulating that noise lifts HEX r on the 25 frames from 0.68 to 0.83 (markers: 0.845), but the gain does not carry to all labels (+0.007).

## Deliveries (dataset index order, 8954 rows; `idx`, `F`, `names`)
- HEX lane: `feats/HEX_input_P2_konan.npz` (10 columns: union set, 3 legible windows + full frame).
  Kept predictions: `out/corr_hex_a100.npz` (`P`, BP row order; only the HEX column changed).
- CV lane: `feats/CV_input_trimmed_konan.npz` (20 columns, trimmed statistics in the predicted box and windows). For a stack if useful; my own read is +0.015.
- All columns: `feats/cellroute_feats.npz`, `feats/trim_feats.npz`, `feats/click_feats_P2_P5_m1_K8.npz`. Mask centroids for all frames: `feats/seg_centroids.npz`.
- Deployability of the kept step needs full-data pre-only detectors for seeds 47 and 48 (only `sg2pre_s46` is deployed).
  It also needs 4-flip TTA, zero-shot Cellpose-SAM masks and CPU geometry. There is no training beyond the ridge.

## 25 marker-frame gate (`source/gate25.py`)
Reference, markers with k=2: CV r 0.899 / LOO-MAPE 7.64; HEX r 0.845 / 7.11.

| Point set (read in the marker hull) | miss | FP | CV k2 r | HEX k2 r |
|---|---:|---:|---:|---:|
| 3-seed TTA peaks >= 0.35 (the CV lane's input) | 0.203 | 0.109 | 0.527 | 0.477 |
| 3-seed TTA peaks >= 0.2 | 0.172 | 0.128 | 0.698 | 0.578 |
| step 1: >= 0.2, merge < 0.5 s, then fill Delaunay holes | 0.171 | 0.125 | 0.778 | 0.475 |
| step 1b: >= 0.2, spacing-prior suppression at 0.7 s | 0.188 | 0.110 | 0.705 | 0.613 |
| step 2: union of 3 detectors and mask centroids, vote >= 1 (P2) | 0.168 | 0.138 | 0.693 | 0.679 |
| step 3: P2 with click-noise emulation (8 draws, 0.16 s) | 0.168 | 0.138 | 0.695 | 0.801 |
| step 3: 5-detector union with click-noise emulation | 0.159 | 0.145 | 0.705 | 0.828 |
| step 4: P2, confident region only | 0.423 | 0.109 | 0.601 | 0.711 |
| markers with random errors at the measured rates | 0.206 | 0.161 | 0.302 | 0.078 |

Region (`source/diag_region.py`), P2:

| Cells read | CV k2 r | HEX k2 r |
|---|---:|---:|
| Predicted technician box | 0.243 | 0.798 |
| Predicted box, trimmed to cell area within 0.4-2.5 × median | 0.739 | 0.716 |
| 150-cell patch | 0.622 | 0.800 |

## All-label reads
Five folds; nested ridge with alpha 100 (fixed) on the log residual of the lane best; one recipe on all folds. Lane bests: CV 10.1142, HEX 9.7380.

| Try (pre-registered from the gate) | Metric | Five folds | vs lane best | f04 |
|---|---|---:|---:|---:|
| P1 cleaned peaks, Konan features | CV | 10.1357 | +0.0215 | +0.008 |
| P2 union, predicted box, trimmed | CV | 10.1296 | +0.0154 | +0.014 |
| **P2 union, windows + full frame (kept)** | HEX | **9.7315** | **-0.0065** | -0.012 |
| P2 union + click-noise emulation (3 and 5 detectors) | HEX | 9.7452 | +0.0072 | +0.001 |
| P2 union, predicted box | HEX | 9.7442 | +0.0063 | +0.002 |
| Gradient boosting on all 144 cell-route columns (different model, one fixed setting) | HEX / CV | 9.9056 / 10.3478 | +0.168 / +0.234 | +0.196 / +0.239 |

## Directions (continue / stop)

| Direction | Status |
|---|---|
| Device rule on detected points, CV | stop: 3 versions worse |
| Device rule on detected points, HEX | stop adding versions. The -0.0065 step is kept for the HEX lane to stack |
| Nonlinear residual model on all cell-route columns | stop: much worse (+0.17 HEX, +0.23 CV) |
| Learned point-set recovery from simulated tessellations (step 3, model version) | not run. Point-set fidelity is not the all-label bottleneck: gate r for HEX rose from 0.48 to 0.83, while residual r stayed near 0 |

# New-recipe inference entry (coordinator job, started 2026-11-10 10:20 EST)

Draft entry: `experiments/lane_cellroute_20261109/inference/`. It does not edit `submission/` or any other lane's files.

| File | Role |
|---|---|
| `recipe_infer.py` | `Recipe(manifest, params).predict(image)`: one raw 972x1296 frame to CD, CV, HEX. `main()` implements the GC interface (one .mha/.tif in, the three json files out) |
| `members.py` | Image-level members: preprocessing (identical to the training loader), recipe A, DINOv3-B, ViT-L; TTA and epoch reads as the lanes' readers |
| `seg_run_feats.py` | HEX lane `seg_run.feats` copied verbatim (seg_run.py starts a run at import) |
| `params_build.py` | Offline: collects or fits every parameter into `params/{fold0,deploy}.json` and writes `manifests/{fold0,deploy}.json`. The entry reads only these files |
| `test_members.py`, `test_fold0.py` | Fold-0 reproduction tests (test harness only; they read stored OOF values to compare and to inject stubs) |
| `audit_inputs.py` | Scan for label / fold / OOF reads on the inference path |

Column formulas called as written by the lanes:
- CD: `cd_combine_20261110.combine` with the CD lane's params (`per_fold['0']` for the test, `deploy` otherwise).
- HEX: `hex_apply.apply` (nested fold-0 steps for the test, deploy steps otherwise), then the cell-route ridge `deploy_hexkonan.correct`.
- CV: the CV lane's apply function, not delivered yet, so CV is a stub.

## Fold-0 reproduction (fold-0 models standing in for full-data weights)

Layer 1 (`test_fold0.py layer1`): the entry's combine stage on the stored per-row inputs, all 1791 fold-0 rows.
- CD max relative difference vs the stored OOF: 0.
- HEX chain: 1.3e-15.
- HEX with the cell-route step: 1.2e-15.
- CV: stub (no CV apply function).

Member parity (`test_members.py`, 3 fold-0 frames, raw image, CPU fp32 vs the stored GPU bf16 predictions):
- recipe A TTA at epoch 20: 0.11 %
- recipe A single view at epoch 8: 0.14 %
- DINOv3-B single view at epoch 16: 0.18 %
- ViT-L TTA at epoch 12: 0.19 %

Layer 3 (`test_fold0.py layer3 3`): frames 0, 3080, 6124, raw image to the final columns, about 15 CPU-min per frame.

| Quantity | Max rel. diff over the 3 frames | Source |
|---|---:|---|
| **CD final** | **0.13 %** | image |
| **HEX chain** | **0.20 %** | image + stubs |
| **HEX final (with cell-route)** | **0.27 %** | image + stubs |
| CV final | not computed | stub |
| A (CV/HEX version) | 0.09 % | image |
| A_cd (CD lane version) | 0.11 % | image |
| D (DINOv3-B, 4-flip TTA, epochs 16-20) | 0.09 % | image |
| V (ViT-L) | 0.43 % | image for the TTA half; the epoch 8-11 half is a stub (fold runs kept final.pt only) |
| P1 (5 Phase-1 members x scalars) | 0.02 % | image |
| Cellpose-SAM mask CD / n_int | 0.22 % / 0.71 % | image (CPU masks) |
| 17 seg features | 15 % (a small count column) | image |
| Image statistics | 0 | image |
| Cell-route features | 81 % (one window column, frame 6124) | image; see below |

Stubs injected from stored values in layer 3: `f6`, `have`, `best`, `graph`, `g_s10x3`, `g_hex46`, `prior_h6c`, `ridge_k3`, `ridge_fall16_h6_s15`, `dino_cvw3`, plus everything CV needs.

Frame 6124 cell-route outlier: the CPU peak set picks a different third legible window. The window region then holds 321 cells instead of 369, and the untrimmed window area CV goes from 0.18 to 0.32. The full-frame features agree within 0.4 %. The window choice is an argmax over peak counts and can flip under tiny numerical changes. This moved that frame's final HEX by 0.27 %.

## Audit: label / fold / OOF reads (`audit_inputs.py`)
Clean, nothing read at inference:
- my modules;
- `cd_combine_20261110.combine`;
- `hex_apply.apply` (its only OOF read is the `oof` command branch, line 79);
- the family builders;
- the green-track peak code;
- the Phase-1 member code.

Violations, all in pieces the recipe needs but that have no per-image version yet:

| File:line | What | Owner (told) |
|---|---|---|
| `lane_hex_20261108/source/feat_prior.py:20` | reads the submission-1 OOF CD (`load_base()['B']`) by frame index for the spacing prior | HEX lane |
| `feat_prior.py`, `feat_ridge.py`, `feat_fuse.py` | `prior_h6c`, `ridge_k3`, `ridge_fall16` built from stored peak files by training index; no per-image entry | HEX lane |
| `lane_cv_20261108/source/graph_train_cv_20261108.py:15,17` | `load_base()` (labels, folds, OOF) and `green_features.npz` at import; REF stretch and CTX from them (lines 19-21) | CV lane |
| `lane_cv_20261108/source/graph_order_cv_20261110.py:17,19` | same | CV lane |
| `lane_cv_20261108/source/tta_infer_20261108.py:14-24` | fold detectors, fold validation indices, `ST.green(0)` (labels) | CV lane |
| `lane_cv_20261108/source/cv_stack5_20261108.py:232` | deploy weights only in the OOF file; deploy stretch not stored | CV lane |
| `meta_ceiling_20261103/source/cell_graph_sweep3_20261105.py:14,16` | GRAPH and `g_s10x3` networks: labels, folds, green_features, CTX at import; no weights saved | no owner (coordinator) |
| `meta_ceiling_20261103/source/sim_embed_eval_20261105.py:11,37` | BEST: labels, folds, green_features, stored GRAPH OOF arrays; leave-one-fold-out ridge only | no owner (coordinator) |

## Detector pair (rehearsal item 8)
- Use the green-track pair `green_track_20261028/deploy/detectors/sg2pre_s47/48.pt` (10-30) everywhere.
- My HEX step's OOF features came from the fold detectors (neither pair). The 11-09 200-frame check used my duplicate pair, which is now moved to `deploy/archive_duplicate_pair_20261110/` (not deleted).
- On 4 CPU frames the green-track pair tracks the OOF features at least as well (median per-frame max rel diff 0.11 vs 0.52).
- The deploy manifest names the green-track pair. Asked the CV lane to confirm its OOF used the fold detectors and its deploy names the green-track pair.

## Deploy smoke test (`test_deploy_smoke.py`, CPU, frame 32 of fold 3)
Every full-data weight that exists loaded and ran end to end in 874 s:
- recipe A s46/s47 epochs 8-20;
- DINOv3-B full-data s46b/s47/s49;
- the provisional Phase-1 five;
- the green-track pre-only detectors;
- Cellpose-SAM.

Stubs: ViT-L (not built), hex6 (GPU-only code), and the pieces without a builder. Not a parity test, since full-data models saw this frame: CD 2382 vs OOF 2417, HEX 0.532 vs 0.525, P1 CD 2524 vs 2510.

## Missing pieces for images -> three columns (deploy), by owner
CV lane:
1. CV apply function and its params file (deploy weights `w_all`, deploy stretch, per-fold versions for the parity test).
2. Per-image graph builder (graph_prep_sp from peaks) and a label-free forward for both graph quartets (full-data weights exist).
3. Frozen heads per image: block statistics from the full-data recipe A s46 tokens, plus the 4 heads (weights exist).

HEX lane:
4. `prior_h6c` per image. Remove the OOF CD read at `feat_prior.py:20`; pass the image's hybrid-B CD, which my entry can provide.
5. `ridge_k3` and `ridge_fall16_h6_s15` per image from peaks. fall16 needs 4 detectors that are not trained; otherwise drop the step (9.7383).
6. `g_hex46`: all-rows mode and weights for `gnn_hex.py`.
7. `dino_cvw3`: full-data CV-weighted DINOv3 s46 keeping epochs 8-12 (2.3-3.4 GPU-h; owner unclear).
8. If `mtcnn5_fix` is kept: CPSAM full data, new `deploy_params.json`, then my cell-route refit.

CD lane:
9. ViT-L s46 full data keeping epochs 8-12 (queued).
10. Phase-1 five full-data members: submission 1's folder `experiments/ssl_release_20260914/model` no longer exists. The manifest uses `region_candidate_20260907/experimental_stage/model` provisionally (same file names; P1 CD 2524 vs OOF 2510 on the smoke frame). Needs confirmation.

No owner (coordinator):
11. GRAPH, BEST and `g_s10x3` have no deployable weights or fit. CV and HEX both use BEST - GRAPH.

Parameters:
12. Two-seed recipe A scalars for CV/HEX are stored by no lane. `params_build.py` fits them on all rows: 0.991413 / 0.989108 / 0.991594. The fold-0 versions reproduce the stored OOF A exactly.
13. hex6 `f6` is wired to the deployed submission-2 corrector (GPU only). Not yet run on GPU.

Container:
14. No Dockerfile for the new recipe. `recipe_infer.main` implements the GC interface (one image in, three json files out). GPU runtime per image is not measured.

## Update 13:15 EST 11-10: CV column, HEX features, CD rules
- CV: `cv_apply_20261110` is wired (h_cv = 0.5 A_cv + 0.5 P1_cv x s1, a_cv = A_cv).
  - Layer 1 with fold-0 weights / sA[0] / stretch0 from `cv_inputs_oof.npz`: CV 5.6e-16 on all 1791 fold-0 rows.
  - The CV lane's deploy params have landed. Its deploy recipe A CV scalar (0.989108) equals the one `params_build.py` fits independently.
  - New CV member: order-loss DINOv3 B/16 s46 (`orderdino_logcv`, single view, epochs 16-20).
- HEX features: `hex_test_features.build_arrays` is wired.
  - pre3 single-view peaks come from the identity flip of the TTA forward (the full SG.heat map). The other 13 detectors run through a device-agnostic copy of `det_peaks_image.load`.
  - With the stored fold-0 peaks and the stored hybrid-B CD it reproduces `prior_h6c` (to 1e-8), `ridge_k3`, `ridge_fall16` and the image statistics exactly (`test_hexfeat_quick.py`).
- CD rules aligned with the CD lane's "Deployment must use" table:
  - V is now the geometric mean of ViT-L s46 and s47 (fold-0 V difference 0.43 % -> 0.008-0.02 %).
  - P1 deploy = the candidate5 package `deploy_review_20260905/staged/model`, x the OOF SCALE.
  - The recipe A snapshot read for CD (`members.recipeA_cd`) was already in place and checked (0.03-0.11 %).

Layer 3 v2 (`logs/test_fold0_layer3_v2.log`; frames 0, 3080, 6124; raw image, fold-0 models, CPU), max relative difference vs the stored OOF:

| Quantity | v2 max |
|---|---:|
| **CD final** | **0.066 %** |
| **CV final** | **0.066 %** |
| **HEX chain** | **0.35 %** |
| **HEX final (cell-route)** | **0.48 %** |
| A / A_cd / D / V / P1 | 0.09 % / 0.11 % / 0.09 % / 0.03 % / 0.02 % |
| DINO_logcv / orderdino_logcv | 0.10 % / 0.26 % |
| mask cd_mm2 | 0.22 % |
| prior_h6c / ridge_k3 / ridge_fall16 | 8.8 % / 11 % (one column) / 0.4 % |
| image statistics | 0 |

The HEX feature differences come from CPU fp32 single-view peaks against the stored GPU bf16 peaks. With the GPU peaks they are exact (`test_hexfeat_quick.py`), and the HEX lane's `det_peaks_image --verify` reproduces the stored peaks exactly on GPU.

Still injected from stored values in layer 3:
- the 13 non-pre-only detector peaks (fold-0 checkpoints exist; skipped to save CPU time);
- f6 / have;
- the GRAPH / BEST / graph-network terms;
- the cvw3 member;
- the CV graph / ridge / quartet / frozen-head terms.

## Missing pieces, revised 13:15 EST
Done since this morning:
- CV apply and its deploy params;
- HEX per-image features;
- the CD snapshot read;
- the ViT-L rule;
- the P1 package.

Still open:
1. CV per-image parts (graphs from peaks, GraphNets for GRAPH and both quartets, SimNets embeddings for the ridge, frozen-head blocks). The CV lane module `cv_image_parts_20261110.py` is in parity testing; I wire it when it passes. It also gives HEX's `graph` / `best` columns (CA.ridge with 'HEX').
2. HEX graph nets `g_s10x3`, `g_hex46` and the `dino_cvw3` member: weights land about 14:00-15:00 EST (cvw3 about 3 h later). I wire them via `gnn_hex_infer.predict` / `family_infer`.
3. Eight new all16 detectors (s51, s52, sigma3p5, sigma5, four frozen heads): queued on GPU 1 from about 13:20 EST.
4. ViT-L s46/s47 full data (CD lane, queued). Manifest paths to be confirmed.
5. `feat_prior.py:20` still reads hybrid_oof at import (numerically unused). The HEX lane has been asked to move it.
6. hex6 `f6` on GPU and the GPU runtime per image: not run yet (no GPU in this job).
7. Container / Dockerfile for the new recipe.

## Update 14:40 EST 11-10: new candidates, CV per-image parts, HEX refit
- **The HEX chain changed** to `chain_mtcnn5fix` (9.7358; new input `cps5`).
  - The kept cell-route step re-read nested on it: **HEX 9.7283 (-0.0075)**, folds -0.017 / -0.010 / -0.009 / +0.002 / -0.003. Deploy params refit (old archived).
- **The CV candidate changed** to the seven-component 10.1052 (order-loss DINOv3 added). The entry reads the rebuilt `cv_inputs_oof.npz`.
- **Layer 1 on the new candidates**, all 1791 fold-0 rows: CD 0, CV 5.6e-16, HEX chain 1.6e-15, HEX with cell-route 1.6e-15.
- **CV per-image parts wired** (`cv_image_parts_20261110`, parity-checked by the CV lane): GRAPH ensemble on single-view graphs; plain and order quartets on 4-flip graphs; sim2_l8 embeddings -> CV and HEX ridge; frozen heads on the recipe A s46 tokens.
  - HEX's `graph` / `best` come from the same parts: the GRAPH HEX column, plus the HEX ridge. Asked the HEX lane to confirm this deploy definition.
  - These parts exist only as full-data weights, so the fold-0 layer 3 keeps them as stubs. They were checked in deploy mode instead.
- **Deploy smoke v2** (`test_deploy_smoke2.py`, frame 32, CPU, 901 s): every CV input from the image with full-data weights.
  - Against the CV lane's deploy in-sample members (stored training peaks, CV column, log): GRAPH 0.0122 vs 0.0097, plain quartet 0.0123 vs 0.0127, order quartet 0.0041 vs 0.0081, ridge 0.0020 vs -0.0008. That is at most about 0.4 % of CV.
  - Expected: the in-sample members used fold-detector peaks; the entry uses the full-data green-track detectors.
  - Output CD 2382, CV 0.397, HEX 0.535.
- `feat_prior.py:20` is fixed by the HEX lane; the audit is clean.

## Missing pieces, revised 14:40 EST
1. ViT-L s46/s47 full data (CD lane; about 22:00 EST 11-10). Paths confirmed.
2. HEX graph nets `g_s10x3`, `g_hex46`, the `dino_cvw3` member and the CPSAM 512 member `cps5` (HEX lane; CPSAM full data ends early 11-11). Wired as soon as names / paths arrive.
3. Two frozen-head detectors of all16 (`frozen_heads/sg2f_s48`, `sg2pref_s46` in the HEX deploy folder).
4. hex6 `f6` and the per-image runtime on GPU (not run; CPU-only job).
5. Container / Dockerfile.

## Update 15:30 EST 11-10: pinning test, HEX graph nets / cvw3 / cps5 wired
- **Pinning test** (`test_pin.py`, the CV lane's suggestion): frame 32's stored fold-detector peaks (single view and 4-flip) fed into `cv_parts`. Against `cv_deploy_insample_members`: GRAPH 2.0e-9, plain quartet 1.1e-9, order quartet 2.4e-9, ridge CV 3.3e-6 and HEX 8.9e-7 (float16 embedding storage).
  - The entry is exact; the 0.0005-0.004 log gaps in the deploy smoke come only from the detector swap (fold detectors vs full-data green-track pair).
- **HEX graph nets wired** (HEX lane "Test-time path"):
  - `g_s10x3` = mean of the HEX columns of the CV lane's `gfull_noctx` and the HEX lane's `full_d64_l5_pre47_s10` / `full_d64_l5_pre48_s10`, run with `gnn_hex_infer.predict` on the graphs `build_arrays` writes.
  - `g_hex46` = `full_hex_pre46_s5` on `g_pre46`.
  - `graph` / `best` confirmed by the HEX lane (CV parts' GRAPH HEX column; plus the HEX ridge).
- **Members `dino_cvw3` and `cps5`** wired: exp of the mean log over epochs 8-12, single view, HEX column. The CPSAM 512 family (centre 512 crop) was added to `members.py`.
  - cps5 fold-0 check from the raw image, fold-0 CPSAM run: 0.045 % / 0.073 % on 2 frames.

Open for deploy:
1. ViT-L s46/s47 full data (about 22:00 EST).
2. HEX `full_d64_l5_pre48_s10` and `full_hex_pre46_s5` graph nets, the cvw3 full-data member, the CPSAM 512 full-data member (early 11-11).
3. Two all16 frozen-head detectors.
4. GPU run (hex6 f6, runtime).
5. Container.

Stubs left in the fold-0 test, all for lack of fold-0 weights or GPU:
- g_s10x3, g_hex46, the CV parts (full-data weights only);
- dino_cvw3 (fold runs kept epochs 16-20, not 8-12);
- f6 (GPU-only code);
- the 13 non-pre-only peaks (fold-0 checkpoints exist; skipped for CPU time).

## Update 15:50 EST 11-10: HEX graph nets landed and pinned
`test_pin_hex.py`: frame 32's stored fold-detector single-view peaks (pre3 + 13) through `Recipe.hex_features`, then `gnn_hex_infer.predict`, against each full-data build's in-sample prediction:
- `full_d64_l5_pre47_s10` (50 weight files): 2.2e-9.
- `full_d64_l5_pre48_s10` (50 weight files): 7.7e-10.
- `full_hex_pre46_s5` (25 weight files): 2.2e-10.

`g_s10x3` and `g_hex46` are therefore live in deploy mode (with the CV lane's `gfull_noctx`, pinned earlier). The HEX lane's empty-graph fix comes in through the import of its current `gnn_hex_infer.py`.

Open for deploy:
1. ViT-L s46/s47 full data (about 22:00 EST).
2. cvw3 full data (about 15:30-16:00 EST) and CPSAM 512 full data (early 11-11), picked up automatically from the manifest paths.
3. Two all16 frozen-head detectors.
4. GPU run.
5. Container.

## Update 15:30 EST 11-10: container draft and file census
- `inference/docker/`: `Dockerfile` (draft; base pytorch 2.11.0-cuda13.0; no `latest`), `requirements.txt` (pinned to the pod venv), `README.md` (build steps), `collect_files.py` (file census).
- Census of one deploy prediction (CPU, file-open hooks): **552 files, 31.1 GB**.
  - Not built yet: ViT-L s46/s47 full data (10 files) and CPSAM 512 full data (5 files).
  - The full-data cvw3 member and all 13 extra detectors now exist and ran.
  - hex6 corrector files are added from the GPU deploy run.
- `members.py`: autocast picks bf16 where the GPU supports it, fp16 otherwise; the repo root comes from `CLEAR_EC_ROOT` (default `/workspace/clear-ec-research/`).

## Update 16:30 EST 11-10: GPU run; detector batch shape matters
- On GPU 3 (coordinator, shared with a ViT-L build; my process 13.4 GB, 4 CPU threads) the HEX features first came out 6-37 % off the stored values, even though they were exact when given the stored peaks. Cause found (`source/diag_peaks_batch.py`, 4 fold-0 frames):
  - The stored peaks were made with batch-4 loaders (`tta_infer_20261108`, `det_peaks_image`). bf16 results depend on the batch shape.
  - The HEX lane's own training-path code reproduces the stored peaks exactly at batch 4 and not at batch 1. The entry's single-image peaks equal that batch-1 output exactly (712 / 1232 / 998 / 1283 peaks).
  - A batch of 4 copies reproduces the stored peaks exactly. Rows are independent, so a batch of 4 different views also does.
- **Fix in `source/deploy_hexkonan_image.py`**: the 4 TTA flips run as one batch of 4; their identity-flip row is the single-view map. The 13 other detectors run on 4 copies.
  - With this, `prior_h6c`, `ridge_k3`, `ridge_fall16` and the cell-route features match the stored values exactly (difference 0).
  - On the first frames HEX final ≤ 3e-5, CD ≤ 6e-5, CV ≤ 2e-4.
  - The cost is extra GPU work for the 13 single-view detectors only; the TTA pass is one batched forward instead of 4.
- The 120-frame fold-0 GPU run restarted with the fix at 16:09 EST (`logs/gpu_fold0_120.log`; about 96 s per frame; ends about 19:20 EST). The 5 frames of the first run are kept in `logs/gpu_fold0_120_batch1peaks_partial.log`.

## Update 19:05 EST 11-10: full-data ViT-L s46 checked (`test_vitl_f99.py`, CPU)
- The CD lane wrote `lane_cd_20261108/arms/vitl16_f99_s46/raw/runs/fold99/dinov3_l16t/` at 18:54 EST. This is the path already in `manifests/deploy.json`.
- sha256 matches the CD lane's list for all 5 files (epoch_08..11, final.pt). Strict load into the `dinov3_l16t` net works.
- My read against the trainer's own stored predictions on its 2 held-out frames (4569, 4570). This is CPU fp32 against the trainer's GPU bf16.

| read | CD max rel. diff | all 3 columns max |
|---|---|---|
| single view, epochs 08-11 | 0.48 % | 0.54 % |
| TTA4, final.pt | 0.05 % | 0.17 % |
| s46 half of V (0.5 mean single + 0.5 TTA, log) | 0.18 % | – |

  - The single-view gap is larger than the TTA gap; averaging 4 flips cancels part of the numeric noise.
  - On GPU with bf16 I expect it to be closer. The deploy GPU run after s47 will show it.
- Sanity check, not parity: on 8 fold-0 frames, the s46 half divided by the stored fold-0 V has median 1.004 (range 0.989 to 1.072). These are different weights, and the full-data model trained on these frames, so a gap is expected.
- Next: when s47 is written, V = geometric mean of the two halves goes live in deploy, then the 20-frame deploy GPU run on GPU 3.

## Update 19:55 EST 11-10: GPU fold-0 reproduction on 120 frames (`test_gpu_fold0.py`, GPU 3)
- Setup: 120 fold-0 validation frames, evenly spaced, run with fold-0 models on GPU, with the batch-shape fix in. The 13 non-pre-only detectors run for real.
- Maximum relative difference against stored fold-0 OOF values:

| value | max | p95 | median |
|---|---|---|---|
| CD final | 0.100 % (1 frame above 0.1 %: 4449) | 0.012 % | 0.003 % |
| CV final | 0.047 % | 0.029 % | 0.011 % |
| HEX final (chain + cell-route step) | 0.014 % | 0.009 % | 0.003 % |
| HEX chain | 0.016 % | 0.011 % | 0.003 % |
| A (CV/HEX) / A_cd | 0.13 % / 0.16 % | 0.06 % / 0.03 % | |
| D / V (TTA half live) | 0.03 % / 0.18 % | | |
| DINO_logcv / orderdino_logcv / cps5 | 0.07 % / 0.16 % / 0.05 % | | |
| P1, mask_cd, HEX features (prior_h6c, ridge_k3, ridge_fall16, cell-route) | ≤ 3.6e-7 | | |

- What is left is bf16 numeric noise in the neural members. Each neural member stays within 0.18 % on every frame. The detector and geometry path is exact.
- Stubbed: 15 inputs that have full-data weights only:
  - f6, have;
  - the 8 CV per-image parts;
  - graph, best;
  - g_s10x3, g_hex46;
  - dino_cvw3.
  
  These are covered by the deploy smoke and pinning tests instead.
- Median time: 126.7 s per frame on GPU 3, which was shared with a ViT-L training run at 100 % utilisation, with 4 CPU threads.

## Update 21:55 EST 11-10: full-data ViT-L s47 checked; V complete
- s47: sha256 matches for 5 of 5 files and strict load works. Checked against the trainer's stored predictions on its 2 held-out frames, CPU fp32 against GPU bf16:
  - single view: CD ≤ 1.05 %. That maximum is one read (epoch 08, frame 4570); the other 7 single-view reads are ≤ 0.35 %.
  - TTA4 final.pt: ≤ 0.19 %.
  - s47 half: ≤ 0.26 %.
- Full V is the geometric mean of the s46 and s47 halves. Against the same rule applied to the trainer's stored predictions, it is off by +0.12 % on frame 4569 and −0.15 % on frame 4570.
- On 8 fold-0 frames, full V divided by stored fold-0 V has median 1.008 (range 0.978 to 1.078). These are different weights, so this is a sanity check only.
- CD now has no stub left in deploy. The deploy GPU run started at 21:44 EST with V live.

## Update 22:30 EST 11-10: deploy-mode GPU run (`test_gpu_deploy.py`, GPU 3)
- Setup: 20 training frames, 4 per fold, evenly spaced. Full-data weights for every member, hex6 corrector live, both full-data ViT-L seeds live.
- Only stub: cps5. CPSAM f99 is not built yet; only epoch_08 is on disk.
- This is not a parity test: the full-data models trained on these frames. The table gives the in-sample deploy value against the stored out-of-fold value:

| column | median | p75 | max (frame, fold) | frames > 2 % |
|---|---|---|---|---|
| CD | 0.44 % | 0.98 % | 2.56 % (4236, f4) | 2 |
| CV | 1.27 % | 1.88 % | 6.00 % (447, f1) | 4 |
| HEX final | 1.30 % | 1.64 % | 4.80 % (4236, f4) | 4 |

- hex6 deploy feature f6 against the stored OOF f6: correlation 0.952, median absolute difference 0.029. The `have` flag agrees on 20 of 20 frames.
- Time: median 74 s per frame on GPU 3, shared with the backbone lane's DINOv3-B run, with 4 CPU threads. The first frame took 173 s, including weight loading.
- `test_gpu_deploy.py` now also stores the signed deploy, OOF and cps5 values per frame, for the rerun once CPSAM f99 lands.

## Update 22:35 EST 11-10: container census on GPU
- `docker/collect_files.py` ran one deploy frame on GPU 3, with hex6 and both ViT-L seeds live. It found 769 files, 48.3 GB. The earlier CPU census, without hex6 or ViT-L, was 552 files and 31.1 GB.
- Not built yet: CPSAM f99 epoch_10, 11, 12 (3 files). epoch_08 and 09 are on disk.
- Biggest folders:

| folder | size |
|---|---|
| lane_cd arms (10 ViT-L files) | 13.9 GB |
| recipe A s46 + s47 | 8.9 GB |
| P1 staged | 7.3 GB |
| lane_hex deploy | 5.6 GB |
| lane_cv arms | 5.2 GB |
| green-track detectors | 4.1 GB |

- Every file holds fp32 weights only.
- **Open question for the packaging owner:** the first submission was 4.9 GB of weights, and its Dockerfile also accepts a GC Model mounted at /opt/ml/model. I have not checked the Grand Challenge limits on image size and model-tarball size. Someone must check them before a 48 GB build.
- If the build has to shrink, one option needs a parity test: store linear and conv weights in bf16 and keep norm parameters in fp32.
  - On GPU, autocast rounds these weights to bf16 anyway, so the matmuls should be unchanged.
  - It would roughly halve the neural members.
- `docker/files.txt` now holds the GPU census. A copy is at `docker/files_gpu_20261110.txt`.

## Update 23:45 EST 11-10: deploy size / runtime budget (coordinator task)
- Full write-up: `inference/docker/size_runtime_budget.md`.
- Size: 769 files, 51.9 GB fp32. ViT-L takes 12.1 GB, recipe A epochs 8-19 8.2 GB, P1 7.3 GB, CPSAM 512 6.1 GB, detectors 5.5 GB.
- Runtime: first frame 159 s, steady 81 s on shared GPU 3, peak GPU memory 17.0 GB. Weight loading takes 29.6 s against 7.7 s of member forwards; HEX features take 20.9 s.
- No-cost savings:
  - the duplicate `cpsam_v2` (1.23 GB);
  - the base weights (1.90 GB). With them replaced by noise, outputs are bit-identical for recipe A, DINOv3-B, ViT-L and a detector.
- Snapshot rules: the lanes average predictions, not weights. ViT-L swa6 tracks already exist for every fold and for full data, so the price of one file per seed can be read from stored predictions.
- Half-precision storage test (fp16, bf16mix) is running on the same 20 frames.

## Update 00:40 EST 11-11: half-precision storage for everything fails CV and HEX
- Test: every float weight is cast to half and back at load, then the same autocast inference runs on the same 20 deploy frames. Change against fp32, median / max:

| variant | CD | CV | HEX |
|---|---|---|---|
| fp16 | 0.013 / 0.032 % | 0.168 / 0.49 % | 0.381 / 1.83 % |
| bf16mix | 0.006 / 0.045 % | 0.158 / 0.84 % | 0.541 / 2.16 % |

- Coordinator rule (median < 0.05 % makes half the default): CD passes, CV and HEX fail.
- The regression members barely move: each member's own output changes by a median of 0.003-0.05 %.
- The thresholded parts move much more. Detector peak counts change by up to 15 per frame, and the hex6 feature f6 by a median of 1.8 %. The graph features built on the peak set and the cell-route features move with them.
- Running now (about 1 h, GPU 3 through the backbone guard):
  - fp16-nodet: detectors, hex6, frozen heads with their recipe A e20 encoder, `cpsam_v2` and the graph models stay fp32. Saves 20.6 GB if it passes.
  - fp16-cdonly: half for ViT-L and recipe A e8-19 only. Saves 10.2 GB.

## Update 01:55 EST 11-11: half precision for the regression members passes; deploy about 28 GB
- fp16-nodet: every regression member is stored in fp16, and the detectors, hex6, frozen heads with their recipe A e20 encoder, `cpsam_v2` and the graph models stay fp32.
  - Change against fp32 on 20 frames, median / max: CD 0.008 / 0.029 %, CV 0.005 / 0.024 %, HEX 0.009 / 0.052 %.
  - That passes the coordinator rule (median < 0.05 %) on all three columns, so it is the default. It saves 20.6 GB.
  - Peak counts, f6, the cell-route features and the graph features are bit-identical to fp32. The detectors alone caused the CV and HEX change seen with fp16 for everything.
- fp16-cdonly (ViT-L and recipe A e8-19 only): CD 0.007 / 0.023 %. CV and HEX stay at the fp32 run-to-run noise level.
- Size with no OOF cost: 51.94 − 1.90 (base weights) − 1.23 (duplicate `cpsam_v2`) − 20.57 (fp16-nodet) ≈ 28.2 GB. Every GB figure for the snapshot rules halves under this default.
- CPSAM 512 full-data, epochs 08-12: all five load strictly. My CPU single-view output matches the trainer's stored predictions within 0.38 % for HEX. The deploy rerun with no stubs started at 06:38 UTC.

## Update 02:20 EST 11-11: deploy run with no stubs
- Every member is live, including CPSAM 512 full-data cps5. 20 training frames over the folds, GPU 3 shared with the CV lane's build. 06:38-07:10 UTC.
- Deploy against stored OOF, median / max of |rel|. This is not a parity test, because the full-data models saw these frames.

| column | median / max | against the earlier run (cps5 stubbed) |
|---|---|---|
| CD | 0.44 / 2.6 % | identical |
| CV | 1.27 / 6.0 % | identical |
| HEX | 1.67 / 7.2 % | was 1.30 / 4.8 % |

- The HEX change comes from cps5. Full-data cps5 against its OOF value: r 0.878, mean log ratio −0.0068. The HEX lane's own 64-frame check gave r 0.954 and −0.0072.
- My cps5 read matches the trainer's stored predictions within 0.38 % (`test_cps_f99.py`, all 5 epochs).
- Time: 92 s median, 165 s for the first frame, on a shared GPU.

## Update 03:15 EST 11-11: CV 6-component freeze in the entry, fold-0 parity passes
- The CV lane changed the freeze candidate by the pre-registered rule: CV 10.1101, 6 components; order-DINOv3 dropped, plain DINOv3 back.
- Entry (`recipe_infer.py`):
  - Plain DINOv3 and order-DINOv3 now run only when the loaded CV weights use them. Under the new parameters order-DINOv3 is never opened, saving 0.86 GB at fp16 and 1.9 s per frame.
  - The CV assembly is one method (`cv_column`), used by `predict` and the tests.
- `params_build.py` rerun:
  - Only the CV block of `params/fold0.json` changed (new w0, stretch0). Every other params and manifests file is byte-identical. The old copy is in `archive/params_before_cv6_20261111/`.
  - The deploy run reads the CV lane's `deploy/` live. Its recipe A and Phase-1 CV scalars equal mine.
- Formula check (CPU, all 1791 fold-0 rows): the entry's CV output equals `apply_fold` exactly, and the stored `cv_oof` within 5.6e-16.
- Live fold-0 check (GPU 3, 120 frames): recipe A, P1 and plain DINOv3 ran with fold-0 weights; the graph, ridge, quartet and frozen-head parts came from the stored inputs.

| quantity | median | max |
|---|---|---|
| CV column | 0.011 % | 0.047 % |
| A_cv | | 0.13 % |
| DINO_logcv | | 0.074 % |
| P1 | ~1e-7 | |

- These match the previous stack's parity.

## Update 04:40 EST 11-11: CD seed audit (two-seed D) in the entry, fold-0 parity passes
- CD lane change (coordinator's seed audit):
  - D = exp(0.5 log D_s46 + 0.5 log D_s47), each seed 4 flips × epochs 16-20.
  - Params file `freeze_cd_params_20261111.json`; at deploy only the stretch changes. Fold 0's nested pick changed too (A 0.45, D 0.10).
  - CD candidate 6.1741.
- Entry changes:
  - `members.dino_cd` accepts per-seed checkpoint lists. A single list gives the old result exactly.
  - `Recipe.cd_column` holds the CD combine.
  - `params_build.py` gives fold 0 the two-seed D (s46 from lane_gpu3, s47 from lane_gpu2, the runs behind the CD lane's TTA files) and the new params.
  - Deploy switches to s46b + s47 and the new params automatically once full-data s47 `complete.json` exists. Until then it keeps s46b alone with the 20261109 params, the coordinator's fallback after 15:00 EST.
  - The old params and manifests are in `archive/params_before_d2_20261111/`.
- Formula check (CPU, all 1791 fold-0 rows): `cd_column` equals the stored `oof_cd` exactly, and so does every fold with its own params. The stored D equals geomean(D46, D47) to 2e-15, and the five-fold CD is 6.1741.
- Live fold-0 check (GPU 3, 120 frames; D live, P1 / A / V / mask stored, each reproduced live on the same frames earlier):

| quantity | median | max |
|---|---|---|
| D | 0.002 % | 0.087 % |
| D, s47 alone | | 0.20 % |
| CD column | 0.0002 % | 0.008 % |

- Full-data s47: epoch 3 at 09:28 UTC, about 3.4 min per epoch, so expected about 10:30 UTC (05:30 EST). A waiter will then run sha256, strict load and the single-view check, and I will rerun `params_build.py`.

## Update 05:40 EST 11-11: deploy D is now the two-seed mean
- The CD lane's full-data DINOv3-B s47 is complete (05:26 EST).
  - The sha256 of epochs 16-20 matches the CD lane's list, and every epoch loads strictly.
- Deploy D on GPU (production path) against the CD lane's reference on the two held images:

| image | two-seed D (mine / ref) | change | s46b | s47 |
|---|---|---|---|---|
| 2019-0064-ODCN | 1082.50 / 1082.03 | +0.04 % | +0.29 % | −0.20 % |
| 2019-0064-OSCN | 1517.17 / 1518.17 | −0.07 % | −0.04 % | −0.09 % |

- The single-seed differences are bf16 noise, the same size as the 120-frame fold-0 parity (s47 max 0.20 %). A CPU fp32 read is 0.1-0.6 % off, so GPU is the reference path.
- `params_build.py` rerun: deploy now has `dino_cd` = s46b + s47, and the CD params equal `freeze_cd_params_20261111.json` deploy exactly.
  - Only `params/deploy.json` and `manifests/deploy.json` changed. The old copy is in `archive/params_before_d2deploy_20261111/`.
- Size +0.86 GB at fp16, so the deploy is about 28.2 GB. D time is about 4.4 s per frame (estimate).

## Update 06:15 EST 11-11: no-stub deploy run with the current stacks
- 20 training frames, every member live, CV 6-component stack and two-seed D (`out/gpu_deploy_20c.json`).
- Deploy against the stored OOF (the new CD and CV OOF files), median / max of |rel|. These are training frames for the full-data models, so this is not a parity number.

| column | now | before (`gpu_deploy_20b`, old stacks) |
|---|---|---|
| CD | 0.51 / 2.4 % | 0.44 / 2.6 % |
| CV | 1.32 / 5.7 % | 1.27 / 6.0 % |
| HEX | 1.67 / 7.2 % | unchanged |

- Time: 92.5 s median on a shared GPU.

## Update 06:30 EST 11-11: HEX seed-audit chain; cell-route refit 9.7262
- The HEX lane's new chain is `chain_seedaudit10` (9.73258; folds 0+4 9.86437): gnn:hex_pre46_s5 is dropped, and cvw3 is the two-seed mean (s46 + s47).
- The cell-route step, re-read nested on it (recipe unchanged, `source/corr_newbase.py`): **HEX 9.72623 (−0.00635)**, folds 0+4 9.8554 (−0.0090). Per fold: −0.017 / −0.008 / −0.008 / +0.001 / −0.001. It replaces 9.7283 on `chain_mtcnn5fix`.
- Entry:
  - The HEX graph net hex46 runs only when the chain has its step, so it no longer runs (−25 weight files).
  - Deploy `dino_cvw3` = family mean over 10 checkpoints (s46 + s47, e08-12).
  - The cell-route params are refit on the new base for fold 0 and deploy.
  - Only the hex blocks, the cell-route params, and the deploy manifest's cvw3 and hex_graph_nets changed. The old copy is in `archive/params_before_hexsa10_20261111/`.
- Layer 1, all 1791 fold-0 rows, all current stacks at once:

| column | max relative difference |
|---|---|
| CD | 0 |
| HEX chain | 1.3e-15 |
| HEX after the cell-route step | 1.2e-15 |
| CV | 5.6e-16 |

- Waiting on the deploy cvw3 s47 build (about 07:00 EST). A waiter then runs the read check and the 20-frame no-stub deploy run.

## Update 07:35 EST 11-11: cvw3 s47 checked; no-stub deploy run with every 11-11 stack
- Full-data cvw3 s47 (complete 06:53 EST): epochs 08-12 load strictly. My single view matches the trainer's stored predictions within 0.22 % for HEX.
  - Deploy `dino_cvw3` (10 checkpoints) on the held frames: 0.4101 and 0.4245. The s46 and s47 halves are 0.418 / 0.402 and 0.432 / 0.417.
- Deploy run, 20 training frames, every member live, CD two-seed D + CV 6-component + HEX seedaudit10 with the refit cell-route step (`out/gpu_deploy_20d.json`).
- Deploy against stored OOF, median / max of |rel| (not a parity number):

| column | median / max | before |
|---|---|---|
| CD | 0.51 / 2.4 % | |
| CV | 1.32 / 5.7 % | |
| HEX | 1.58 / 7.0 % | 1.67 / 7.2 % on the old chain |

- Time: 92.4 s median on a shared GPU.

## Update 09:55 EST 11-11: budget on the current stacks; fp16 copies written
- The coordinator asked for the budget in one message (sent 08:40 EST). They approved the fp16 copies on the condition median < 0.05 % per column.
- Re-measure on the current stacks (CD two-seed D, CV 6-component, HEX seedaudit10, real cps5):
  - fp16-nodet against fp32, median / max: CD 0.009 / 0.027 %, CV 0.005 / 0.021 %, HEX 0.006 / 0.030 %. **Pass.**
  - The fp32 run reproduces the no-stub deploy run 20d (CD identical, HEX 4e-9, CV 1e-8).
  - Runtime: steady 90.0 s, first frame 213 s (cold reads 43.7 s). Per stage: section 4b of the budget document.
  - The HEX graph nets took 17.1 s for 100 nets, against 9.2 s for 125 yesterday. I have not explained this; CPU contention is suspected.
- fp16 copies (`docker/convert_fp16.py`, gate checked):
  - 77 files, 42.85 GB → 21.43 GB, in `deploy_fp16/`. Every copy is bit-identical to the simulated cast, and nothing was overwritten.
  - Pointing the entry at them waits for the PI's packaging word.
- Size, current stacks: 53.64 GB fp32 (749 files). With the savings it is ≈ 29.1 GB.

## Update 10:30 EST 11-11: joint nested HEX ridge; cell-route refit and strictness
- The HEX freeze is now the joint nested ridge (`chain_joint`: 9.72621 five folds, 9.85123 folds 0+4; entry `hex_apply.hex_any`, `deploy_params_joint.json`).
- Cell-route step, kept recipe refit (weights nested): **9.71154 (−0.0147)**, folds 0+4 9.84609 (−0.0051). Per fold: −0.023 / −0.028 / −0.026 / −0.009 / +0.013.
- Strictness (asked by the coordinator):
  - Nested: the standardisation, the ridge weights and c / c0.
  - Fixed in advance: alpha 100, ≥ 50 cells (the step's only on/off rule) and the feature thresholds. There is no hex6 rule in the step.
  - Not nested: the feature version was chosen on all-label five-fold reads of 15 versions, and the point sets were pre-registered from the 25-frame expert-marker gate.
  - Strict re-read (version choice incl. "no step" nested by inner 4-fold CV on the training folds): **9.72169 (−0.0045)**, folds 0+4 9.85786 (**+0.0066**). Per fold: +0.0005 / −0.0283 / +0.0014 / −0.009 / +0.0128. Fold 0 picked P2 + creg and fold 2 picked P3.
  - Without the 25 gate frames: kept −0.0094, strict +0.0003.
  - Decision to the coordinator.
- Entry wired to `hex_any` (lane params embedded; fold 0 = nested[0]).
  - g_hex46 and the hex46 graph nets are back, cvw3 is two-seed, and the cell-route params are refit on `chain_joint`.
  - `HEX_FORM=chain` falls back to the seed-audit chain. An optional `hex_extra` manifest hook covers a later attention-pool column.
  - Layer 1 (1791 fold-0 rows): HEX chain 4.4e-16, HEX cell-route 4.4e-16, CD 0, CV 5.6e-16.
  - The old params are in `archive/params_before_hexjoint_20261111/`.
- The 20-frame no-stub deploy run is running (launched 10:17 EST).
- **Coordinator decision (10:40 EST):** keep the step with the kept feature version; deploy does not switch versions per fold.
  - Quote the strict value: **HEX 9.7217 (−0.0045 against the joint ridge 9.7262)**. Outside the 25 gate frames the strict gain is about zero.
  - Both numbers stand: **9.7115 as computed** (version chosen on all-label reads) and **9.7217 strict**.
  - The `HEX_FORM=chain` fallback and the optional attention-pool hook stay.
  - Refit once more (as computed and strict) when the HEX lane switches to the 39-column ridge with the backbone lane's verified attention-pool build.
- No-stub deploy run with the joint HEX (`out/gpu_deploy_20e.json`, 20 training frames).
  - Deploy against stored OOF, median / max: CD 0.51 / 2.4 %, CV 1.32 / 5.7 % (both unchanged), HEX 2.05 / 4.5 %. The step-chain run 20d gave 1.58 / 7.0 %.
  - The hex46 graph nets ran. 92.8 s median per frame.
  - Not a parity number; fold-0 layer 1 is exact.

## Update 11:55 EST 11-11: 39-column HEX ridge with the attention-pool member
- The HEX freeze is now the joint ridge with `attnp5`: `chain_joint_attnp` 9.71882, folds 0+4 9.84826 (`deploy_params_joint_attnp.json`). `attnp5` is the backbone lane's AttnPool DINOv3-B f99, raw epochs 8-11 + final, log mean of HEX.
- Cell-route step on it:

| | five folds | folds 0+4 | without the gate frames |
|---|---|---|---|
| as computed | **9.70581 (−0.0130)** | 9.84236 (−0.0059) | −0.0073 |
| strict (version choice nested) | **9.71576 (−0.0031)** | 9.85371 (+0.0055) | +0.0021 |

- Per fold: as computed −0.027 / −0.020 / −0.026 / −0.008 / +0.015; strict −0.004 / −0.020 / +0.001 / −0.008 / +0.015.
- Entry:
  - `HEX_FORM=joint_attnp` is the default; the joint and chain fallbacks stay.
  - `attnp5` comes in via the `hex_extra` hook and `members.attnpool_hex`, which builds the backbone lane's AttnPool network and runs the forward as their `evaluate()`.
  - The cell-route params are refit on `chain_joint_attnp`. The old params are in `archive/params_before_hexattnp_20261111/`.
- Layer 1, all 1791 fold-0 rows: HEX chain 4.4e-16, HEX after the cell-route step 4.4e-16, CD 0, CV 5.6e-16. OOF `attnp5` is read as `chain.cnn5_hex`.
- AttnPool smoke (GPU):
  - sha256 matches the hand-over (5/5).
  - My per-checkpoint read against the trainer's stored held-row predictions: HEX ≤ 0.25 % (bf16, batch 1 against 8).
  - The SSL weights at build give bit-identical `attnp5`, so the SSL file is not needed.
  - `attnp5` on the held rows: 0.4046 / 0.4279.
- Running: the 20-frame no-stub deploy run, then the budget re-measure (fp32 + fp16-nodet).
- 12:30 EST: the chain finished.
  - No-stub deploy run 20f (39-column HEX with `attnp5`), deploy against stored OOF, median / max: CD 0.51 / 2.4 %, CV 1.32 / 5.7 %, HEX 1.63 / 4.6 %; 97.0 s median.
  - Budget fp32: steady 98.1 s, first frame 166 s; `attnp5` costs 2.0 s (3.2 s on the first frame). The HEX graph nets are now 21.3 s (125 nets).
  - fp16-nodet against fp32: CD 0.009 / 0.027 %, CV 0.005 / 0.021 %, HEX 0.007 / 0.026 %. Pass, so the 5 `attnp5` fp16 copies were written: 1.769 → 0.885 GB, bit-identical.
  - Size: 779 files, 55.42 GB fp32; ≈ 30.0 GB with the savings. `deploy_fp16/` holds 82 files, 22.31 GB.

## Update 14:25 EST 11-11: framework revision; this lane is the DEPLOY lane
- The shared ledger (`ledger_20261111.jsonl`) is now the channel. Decisions carry ids, and lessons L1-L7 gate launches (`docs/FRAMEWORK_REVISION_20261111.md`, local repo copy).
- The deploy set implements **D-CD-001, D-CV-001, D-HEX-004 and D-HEX-005**. The ids are written into `params/*.json` and `manifests/*.json` under `decisions`.
- Conformance check (`check_l5_decisions.py`): every item is true.
  - CD params equal `freeze_cd_params_20261111` deploy, with D = s46b + s47.
  - CV reads the lane deploy folder (md5 3b437500, 6 components, no orderdino).
  - The embedded HEX params equal `deploy_params_joint_attnp.json` (md5 2a2d8062, 39 weights, `attnp5`, hex46, cvw3 two seeds).
  - The cell-route step is the kept version on `chain_joint_attnp`.
- **L5 for the cell-route step:** `corr_newbase.py`'s fit run with no outer fold reproduces `params/cellroute_deploy.json` exactly (all diffs 0.0, 8825 rows, apply 0.0).
- Ledger records appended:
  - A-DEPLOY-001: deploy set with md5s;
  - A-DEPLOY-002: size / runtime table;
  - A-DEPLOY-003: fp16 copy set, with sha256 list `docker/deploy_fp16_sha256.txt`;
  - O-DEPLOY-001 to 006: conformance, L5, layer 1, deploy run 20f, fp16 parity, refit values;
  - F-DEPLOY-001: the strict cell-route value's convention.
- **Convention note (L6):** the strict 9.71576 is on the HEX lane's nested `chain_joint_attnp` base (9.71882). It is not stacked on their fully strict 9.71977. The end-to-end strict number has not been computed.

## Update 15:05 EST 11-11: end-to-end strict HEX with the cell-route step
- The coordinator asked for it after F-DEPLOY-001. Method (`source/corr_strict_e2e.py`): my strict procedure on the HEX lane's strict fold runs (`runs/strict/strict_f*.npz` joint_attnp, assembled with `chain.final`).
  - My strict procedure: version choice among 15 + none by inner CV on the training folds; weights and scales on the training folds.
  - Incumbent recomputed in the same run: 9.71977 / 9.84826.
- Results against the HEX strict 9.71977:

| base convention | five folds | folds 0+4 | without gate frames |
|---|---|---|---|
| (a) chain: run f's stretched candidate on every row, no fold-f label anywhere, fully strict | **9.71342 (−0.0064)** | 9.84958 (+0.0013) | −0.0036 |
| (b) OOF: assembled strict OOF on every row, deploy-like, second-hand leak | 9.71663 (−0.0031) | 9.85366 (+0.0054) | +0.0021 |

  - Per fold (a): −0.0044 / −0.0160 / −0.0206 / +0.0022 / +0.0071.
- For the verifier's integrated table: 9.71342 (A-DEPLOY-004).
- L5 for the strict procedure (answers F-VER-001): its version choice with no outer fold picks the kept version (9.70581; second P2 + creg 9.70788). So the deploy parameters are the strict procedure with no outer fold (O-DEPLOY-008).
- The HEX strict OOF can be assembled from the HEX lane's fold runs and is stored in my npz as `base` (O-DEPLOY-009, pointer for Q-VER-001).

## Update 22:35 EST 11-11: Fourier cell-spacing CD column (D-CDSIMPLE-001 a)
- Method (`source/fft_spacing.py`), one fixed form, no labels used:
  - Hann-windowed power spectrum, radial mean.
  - One quadratic 1/f fit in log-log over 0.006-0.25 cycles/px, refit once without ±25 % around the first peak.
  - Band peak (500-5000 cells/mm²) with a parabolic refinement; D = (√3/2)(f0/s)²·10⁶ with s = 1000/1296 µm/px.
  - Full frame and 70 tiles of 256 px. The coordinator checked the 6-frame probe (`out/fft_spacing_probe.png`) before the full run.
- All 8,954 frames, 0 errors, 108 s on 15 CPU workers: `feats/fft_spacing_cd.npz`.

| column | log r all | P1 | P2 | on cd_mm2 rows | nested MAPE alone |
|---|---|---|---|---|---|
| FFT full frame | 0.568 | 0.503 | 0.675 | 0.636 | 10.87 |
| FFT tile median | 0.676 | 0.605 | 0.775 | 0.733 | 9.59 |
| FFT legible-tile median | 0.647 | 0.589 | 0.763 | 0.703 | 9.81 |
| cd_mm2 (86 % of rows) | 0.803 | 0.791 | 0.867 | 0.803 | 7.37 |
| detector density | 0.657 | 0.549 | 0.732 | 0.733 | 9.72 |

- L1 against the frozen CD stack: the tile median's residual correlation is 0.66, but its deviation from the stack predicts the stack's error with r = 0.02 (cd_mm2: 0.002). So it carries no new information; it is only noisier.
- The within-frame tile spread correlates 0.21-0.26 with the stack's absolute error, a difficulty signal.
- Ledger: A-DEPLOY-005, O-DEPLOY-010, O-DEPLOY-011. The verifier is asked for the nested stack read (Q-DEPLOY-001).

## Update 23:55 EST 11-12: T4 path (D-PKG-004 / D-PKG-006 / D-GPU-001), real-T4 check, four packaging bugs

**Real T4 (the PI's Colab runtime: Tesla T4 15 GB, torch 2.11.0 cu126, 2 CPUs, 12 GB RAM, disk ~400-500 MB/s).** Staged t1 package laid out as
Dockerfile.freeze; every job = one image in a fresh process through gc_entry.py with the fixed image environment; frames 0, 447, 3348 of the 20.

| tier, fp16 + 1 detector copy (container default on a T4) | wall per job, cold disk | reads in it | warm estimate | GPU peak alloc / device | RAM | strict CD / CV / HEX (mean) |
|---|---|---|---|---|---|---|
| t1 | 323 / 323 / 332 s | 72-79 s, 28.9 GB | ~262 s | 4.31 / 4.33 GB | 5.7 GB | 6.1737 / 10.1115 / 9.71342 (8.6662) |
| p | 287 / 288 s | 55-60 s, 24.6 GB | ~237-242 s | 4.31 / 4.33 GB | 5.0 GB | 6.1737 / 10.1115 / 9.71943 (8.6682) |
| t2 | 273 / 277 s | 48 s, 23.4 GB | ~233-238 s | 4.31 / 4.33 GB | 3.1 GB | 6.1875 / 10.1115 / 9.71943 (8.6728) |
| t3 | 173 / 174 s | 26 s, 13.9 GB | ~152 s | 4.31 / 4.33 GB | 3.0 GB | 6.2138 / 10.1138 / 9.73076 (8.6861) |
| t1, fp16 with 4 copies | 376 / 398 / 376 s | 70-83 s | ~315 s | 4.31 / 4.33 GB | 5.7 GB | |
| t1, all-fp32 / mix | killed at 600 s, not finished | | | 4.6 GB device | | |
| t1, native bf16 | OOM after 15 s: no bf16 attention kernel on sm75, the math path asks for 18.5 GB | | | | | |

- Cold: Colab's 12 GB RAM cannot cache a job's reads, so every Colab job read from disk. Warm estimate = wall - Colab read time + page-cache
  read at 2.8 GB/s (pod measurement): valid only if Grand Challenge keeps the archive in its 32 GB page cache between jobs. The 2 CPUs cost
  little: the CPU stages (HEX features, graph nets, cell-route) are ~15 s of a job. Container start and archive mount on the platform are not in
  these numbers.
- t3's CV values on Colab are timing-only (t1's CV parameter files in place of t3's, which differ in one stacking weight; same components).
- The real T4's outputs agree with the H200 emulation of the same path (3 frames: CV 0.04 %, HEX 0.07 % median), far closer than either is to
  the bf16 reference, so the H200-measured parity and impact below carry over.

**Parity and score impact (H200 emulation of the T4 paths, TF32 off; reference = original fp32 weights, bf16, the lanes' numerics; t1).**
Median / max deviation per metric; impact = the run's per-frame deviations applied to the strict t1 OOF (8,954 rows), score = mean of the three.

| path | frames | CD | CV | HEX | peak sets identical | strict-score impact (95 %) |
|---|---|---|---|---|---|---|
| fp16 + 1 copy (T4 default) | 200 | 0.035 / 0.21 % | 0.175 / 0.83 % | 0.44 / 3.9 % | - | +0.0066 (+0.0016 to +0.0121) |
| fp16 + 1 copy | 20 | 0.038 / 0.13 % | 0.168 / 0.99 % | 0.355 / 1.74 % | 0/420 | +0.0065 (+0.0014 to +0.0119) |
| fp16, 4 copies | 20 | 0.038 / 0.13 % | 0.168 / 0.99 % | 0.355 / 1.74 % | 0/420 | +0.0066 |
| fp32 only in detectors / hex6 (mix) | 20 | 0.038 / 0.13 % | 0.223 / 0.92 % | 0.437 / 1.54 % | 0/420 | +0.0053 |
| all-fp32 (1 or 4 copies) | 20 | 0.038 / 0.13 % | 0.221 / 0.91 % | 0.436 / 1.54 % | 0/420 | +0.0053 |
| bf16 emulated in fp32 arithmetic | 20 | 0.016 / 0.19 % | 0.092 / 0.59 % | 0.417 / 1.94 % | 0/420 | +0.0062 |
| native bf16, FlashAttention-2 (A10G proxy) | 20 | 0.011 / 0.06 % | 0.197 / 0.61 % | 0.426 / 1.66 % | 20/420 | +0.0059 |

- Measured directly on the 200 labelled rows (deploy weights): score change -0.006 (-0.040 to +0.028); 200 rows cannot resolve 0.006.
- No path meets the 0.05 % rule, the A10G proxy included: thresholded detector peaks flip under any kernel change (not one of 420 peak sets
  survives even fp32), and the bf16 reference itself was computed with cuDNN attention, which only the H200 runs. fp32 is not closer to the
  validated outputs than fp16 (+0.0053 vs +0.0065, inside each other's intervals) and does not finish in 600 s on the T4.
- One copy instead of 4 identical copies in the 13 detectors: outputs change by < 5e-7 relative (fp16 and fp32), 0.73 of 1.9 PFLOP saved,
  124 -> 58 s on the T4.
- FLOPs per t1 frame 1.9 PFLOP; real T4 fp16 throughput on the detectors ~5-9 TFLOPS (the FLOP-based estimate assumed 15-40 and was optimistic).

**Packaging bugs found by the real-T4 run (none could show on the pod).** (1) PYTHONPATH (runtime folder) missing in the image: cellpose's
CPDINO import failed silently, every image got the fallback constants. (2) stage_package's test-file filter dropped timm's
layers/test_time_pool.py. (3) CLEAR_EC_UNLOAD not set in the image: ~17 GB GPU peak, OOM on the T4. (4) A missing CV parameter folder silently
gave constants (recipe_infer sets cvp=None). Fixed in Dockerfile.freeze (ENV PYTHONPATH, PYTORCH_CUDA_ALLOC_CONF, CLEAR_EC_UNLOAD), gc_entry.py
(runtime path first, unload default, loud start-up checks for the CPDINO import and the CV parameters), stage_package.py (runtime packages
dinov3 / timm whole, filter limited to the lane's tests); all five contexts restaged. pack_hooks.py: below compute capability 8.0 the default is
fp16 + one detector copy.

**Recommendation for the T4: build p first.** Largest tier that finished inside 300 s on the real T4 with cold reads on a slower disk (287-288 s;
~240 s warm); strict mean 8.6682 vs t1 8.6662 (+0.002, inside the noise). t1 took 323-332 s cold on the T4 and fits only if the platform reads
the archive at least ~2x faster than Colab or warm. t2 buys only ~12 s over p for +0.005; t3 (~174 s) is the fallback if p times out.

## Update 20:45 EST 11-13: build box images, archives and Grand-Challenge-style tests (p and t3)

- **Upload files** are in `build box ~/workspace/clear-ec/package_20261112/upload_20261113/` (see `SHA256SUMS.txt` and `UPLOAD_TRYOUT_KO.md`):

  | tier | image id | image archive | model archive |
  |---|---|---|---|
  | p | dfc08a448e58 | 4.11 GB | 15.06 GB |
  | t3 | 8853d77c1f5a | 4.11 GB | 8.81 GB |

  Ledger: A-DEPLOY-012 and A-DEPLOY-013.
- **Three packaging bugs found and fixed** (F-DEPLOY-005). The test ran the image as uid 10001 with the extracted archive, which no earlier test did.
  - A mode-600 code file (`phase2_lejepa_20261003.py`) gave fallback constants on every image.
  - `cpsam_v2` is mode 600 in the model tree.
  - The t2 and t3 requirements lacked timm, so the t3 build failed.
  - Fixes are in `Dockerfile.freeze` (chmod plus a build-time readability check as uid 10001), in `stage_package.py` (timm pinned for every tier) and in the archive script (`--mode=a+rX`).
  - Contexts were restaged (A-DEPLOY-011).
- **RTX 4090 tests** (O-DEPLOY-022 and O-DEPLOY-023): 17 of 17 jobs passed per tier, on the default path and on the forced T4 path.
  - The forced T4 path matches the real Colab T4 to CD ≤ 0.005 %, CV ≤ 0.14 % and HEX ≤ 0.17 %.
  - Its outputs on the three platform debug cases are the Try-out expectations.
- **Risk** (F-DEPLOY-006): the p model archive (15.06 GB) may exceed the 10 GB per-upload cap that several Grand Challenge pages state. The t3 archive fits.
- **Scripts:** `inference/buildbox_20261113/`.
