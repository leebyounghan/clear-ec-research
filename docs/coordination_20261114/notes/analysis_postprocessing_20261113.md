# Post-processing and non-network algorithms in the frozen candidate (tier p), 2026-11-13

Scope: everything between the networks' raw outputs and the three submitted numbers, read from the inference code and the shipped parameter files on pod `clear-ec-research-4gpu-1104`. Read-only. Measured facts only; "not found" where the records do not give a value.

## 0. Conventions and what tier p actually ships

Path abbreviations (all under `/workspace/clear-ec-research/experiments/` unless they start with `/`):

| Short | Path |
|---|---|
| INF | `lane_cellroute_20261109/inference` |
| CRS | `lane_cellroute_20261109/source` |
| LCD / LCV / LHX | `lane_cd_20261108/source`, `lane_cv_20261108/source`, `lane_hex_20261108/source` |
| MC | `meta_ceiling_20261103/source` |
| GT | `green_track_20261028` |
| P1APP | `submission1_test_20261030/source/phase1_app` |
| PP | `INF/params/deploy_p.json` (entry parameters of tier p) |
| CRP | `INF/params/cellroute_deploy_p.json` |
| CDP | `lane_cd_20261108/reads/freeze_cd_params_t1_20261112.json` |
| CVP | `lane_cv_20261108/deploy_t1/cv_deploy_params.json` + `cv_ridge_params.npz` |
| HXP | `lane_hex_20261108/preds/deploy_params_joint_t2.json` |
| H6P | `GT/deploy/hex6_blend_params.json` |

Ledger = `ledger_20261111.jsonl` (ids in brackets). "Strict" = the scored fold's labels hidden in the whole procedure; "as computed" = the lane's nested stack as built. MAPE in %.

Facts about tier p that differ from the short description of the candidate:

- **Tier p = CD t1 + CV t1 + HEX t2 + cell-route refit on the t2 OOF** (`PP` key `decisions`; `INF/params_build.py:137`). The image copies `deploy_p.json` over `deploy.json` at build (`INF/docker/Dockerfile.freeze:33-37`); the entry then reads `manifests/deploy.json` and `params/deploy.json` (`INF/docker/gc_entry.py:49-50`; `CLEAR_EC_TIER` is not set in the image).
- **The HEX ridge in tier p has 38 columns, not 39.** `HXP` lists `dropped: ["mtcnn5"]` and its `deploy.w` has 38 entries. The 39-column ridge (D-HEX-004) is tiers full and t1.
- **The CD blend weights in tier p are P1 0.25 / A 0.40 / D 0.10 / V 0.25** (`PP.cd.weights`), not the 0.25 / 0.45 / 0.10 / 0.20 of D-CD-001. Recipe A is epoch 20 TTA only and ViT-L is final.pt TTA only (`CDP.inputs`).
- **The CV DINOv3 term in tier p is one seed (s46b), not three** (`lane_cv_20261108/deploy_t1/README_tier.json`, key `dino_term`).
- `PP.hex.P` equals `HXP` (checked in this analysis: Python dict equality True). `PP.cd` equals the `deploy` block of `CDP` (all digits equal).
- Strict five-fold values of tier p: CD 6.1737, CV 10.1115, HEX 9.71943, mean 8.6682 [O-DEPLOY-018, D-PKG-003]. Best pipeline with scalars refit per fold: 6.26103 / 10.14786 / 9.79992 / 8.73627 (`docs/BRIEF.md:78`).

Row counts used by the fits (read from `MC/best_pipeline_20261106.py` at run time): 8,954 rows; `good` (not `seen_leaked`) 8,934; rows with a usable hex6 feature 8,927; HEX label > 0 on 8,952.

## 1. Shared stages (inputs of more than one metric)

### 1.1 Image preprocessing

| Consumer | What | Code |
|---|---|---|
| Regression members (recipe A, DINOv3-B, ViT-L, cvw3, attention pool), detectors, hex6 | CLAHE clip 2.0 with a (40, 30) tile grid on the raw 972x1296 uint8 frame, then 1-99 percentile scaling clipped to [0, 1] | `INF/members.py:23-31`; `GT/deploy/hex6_correction.py:34-40` |
| Family inputs | 3 identical channels; replicate pad right 16 / bottom 20; ImageNet mean/std only for `dinov3_b16` and `dinov3_l16t` | `INF/members.py:19-20, 34-40` |
| Phase-1 members | Central 512x512 crop; five 256x256 tiles at (0,0), (0,256), (256,0), (256,256), (128,128); per tile CLAHE 2.0 (8, 8) and 1-99 percentile scaling | `P1APP/final_candidate_core.py:12-13, 21-34, 37-45` |
| Cellpose-SAM segmentation | CLAHE clip 3.0, tiles (8, 8) on the raw frame | `INF/recipe_infer.py:55` |
| Image statistics (HEX ridge) | Raw frame at half size: mean, SD, share of pixels < 5, variance of the 4-neighbour Laplacian | `INF/recipe_infer.py:32-35` |

### 1.2 Member read rules (TTA, checkpoint and seed averaging)

Every regression member outputs standardised logs z; the log prediction is `z * log_std + log_mean` with the checkpoint's own constants (`INF/members.py:77`). Flips are the four exact flips (none, horizontal, vertical, both), `INF/members.py:17, 70-73`.

| Member | Averaging | Forwards per image | Code | Manifest key (`INF/manifests/deploy_p.json`) |
|---|---|---:|---|---|
| P1 (Phase-1 five crop members) | Per member: mean of z over 40 views (5 tiles x 8 dihedral views), then `exp(z*LS+LM)` with fixed `LM`, `LS`; arithmetic mean over the 5 members | 200 | `P1APP/final_candidate_core.py:10-11, 37-45, 92-102`; `INF/recipe_infer.py:59-67` | `p1_members` (5 files) |
| A (recipe A) | Per seed (46, 47) at epoch 20: mean z over 4 flips, exp; arithmetic mean over the 2 seeds | 8 | `INF/members.py:97-99` | `recipeA_tta` (2 files) |
| D (DINOv3-B/16) | Per seed (s46b, s47): mean log over 4 flips x epochs 16-20; mean of the two seed logs; exp | 40 | `INF/members.py:112-117` | `dino_cd` (2 x 5 files) |
| V (ViT-L/16) | Per seed (46, 47): mean log over 4 flips at final.pt (epoch 12); mean over seeds; exp | 8 | `INF/members.py:125-128` | `vitl_seeds` (`tta` only; `single` empty) |
| DINO term of CV | s46b, epochs 16-20, single view, mean log | 5 | `INF/members.py:120-122` | `dino_cv` (1 x 5 files) |
| cvw3 (HEX column) | Seeds 46 and 47, epochs 8-12, single view: exp of the mean log over the 10 checkpoints | 10 | `INF/members.py:177-180`; `INF/recipe_infer.py:175` | `cvw3` (10 files) |
| attnp5 (HEX column) | Seed 46, raw epochs 8-11 + final.pt, single view: exp of the mean log over the 5 checkpoints | 5 | `INF/members.py:162-174` | `hex_extra.attnp5` (5 files) |

Measured effect of the averaging choices, CD stack, five folds (`final_summary_cd_20261113.md` section 4a): four-flip TTA over epochs 16-20 for family members -0.008 against a single view (11-08); ViT-L seed mean -0.0049 (11-09); DINOv3 two-seed mean +0.0016, adopted by the seed-audit rule [O-CD-001]; recipe A epoch 8-19 snapshot -0.004 as computed and zero under strict nesting [O-CD-TIERS], and tier p does not ship it.

### 1.3 P1 SCALE and recipe A scalars

- **P1 SCALE** = [0.9970446110788349, 0.987972130802936, 0.9892234872488541] for CD, CV, HEX (`PP.P1_scale`; hard-coded at `INF/params_build.py:36`, `MC/ceiling_meta_cpu_20261103.py:27`, `recipeA_analysis_20261028/source/hybrid_oof_20261028.py:19`). Applied as `mean of members * SCALE` (`INF/recipe_infer.py:67`). Origin per the repo: the MAPE-minimising weighted median of label / prediction on the 7 September five-fold OOF, all folds, not refit afterwards (`code/reporting/build_timeline.py:41`; `docs/legacy_phase1/METHODOLOGY.md:44-50`). The script that fitted it: not found.
- **Recipe A scalars** = [0.9914125837684035, 0.9891082957404466, 0.9915938787867987] (`PP.A_scalars`). Weighted median of y/p with weights p/y over rows with y > 0, fitted on all rows for deployment (`INF/params_build.py:51-54`; `PP.A_scalars_note`). The OOF values used the leave-one-fold-out scalars of `hybrid_oof.json`; the all-row recipe with an outer fold reproduces them exactly [O-CV-L5-001].
- CD in tier p re-fits both scalars inside its own procedure (section 2). CV and HEX use the two vectors above as they are.
- Measured size of nesting these scalars: best pipeline as computed 8.73923 against 8.73627 with P1 SCALE and recipe A scalars refit per outer fold (CD 6.26387 / 6.26103, CV 10.15119 / 10.14786, HEX 9.80262 / 9.79992), `docs/BRIEF.md:77-78`.

### 1.4 Cellpose-SAM zero-shot masks and their 17 features

- `cpsam_v2`, `flow_threshold 0.8`, `cellprob_threshold -2.0`, batch 8, on the CLAHE frame (`INF/recipe_infer.py:48-57`).
- `feats(mask)` (`INF/seg_run_feats.py:9-29`, a verbatim copy of `LHX/seg_run.py` lines 20-46): neighbour pairs within radius 3 px; a cell counts as interior ("ok") if it does not touch the frame edge, its background share of the dilated ring is < 0.25 and its area > 30 px. Output order: `h6, h5, h7, n_int, area_cv, cd_mm2, coverage, n_cells, n3..n9, h6_strict, n_strict`.
  - `n_int` = number of ok cells (index 3). `cd_mm2 = 1e6 / (mean ok-cell area in px * (1000/1296)^2)` (index 5; `INF/seg_run_feats.py:5, 28`).
  - Returns all-NaN when the mask has < 10 labels or < 5 ok cells (`:11, :24`).
- Centroids and areas of the masks feed the cell-route step (`CRS/deploy_hexkonan.py:67-73`).

### 1.5 Detector peaks

- Three pre-only detectors `sg2pre_s46/47/48` (manifest `preonly_detectors`). Each runs the 4 exact flips as one batch of 4; the heat maps are un-flipped on the in-frame cells and averaged (`CRS/deploy_hexkonan_image.py:37-49`).
- Peak finder `SG.peaks2` (`GT/source/green_sg2_specialist_20261028.py:79-107`): on the stride-2 probability map (grid 496x656, `:31-33`), local maxima of a 5x5 window with probability >= threshold inside the frame support, position refined by the probability-weighted 3x3 centroid, then greedy NMS at 12 px (`NMS2 = 12.0`, `:35`). Stored threshold 0.2.
- Three peak sets per image (`CRS/deploy_hexkonan_image.py:57-61`): peaks of the mean TTA map of the 3 detectors ("seeds3_tta4"), per-detector TTA peaks, per-detector single-view peaks (identity flip of the same batch).
- 13 further detectors for the 16-detector consensus (9 sg2 specialists and 4 frozen heads on the recipe A s46 encoder; `INF/recipe_infer.py:24-25, 74-83`): single view, run as a batch of 4 identical copies (`CRS/deploy_hexkonan_image.py:81-85`), or n copies under `CLEAR_EC_DET_COPIES` (section 8).
- Thresholds applied downstream: 0.2 (stored peaks, consensus candidates), 0.35 (graph nodes, hex6, cell-route sources), 0.5 (window selection, "legible" peaks). Where and on which rows 0.2 / 0.35 / 0.5 and the 12 px NMS were chosen: not established in this analysis.

### 1.6 Window selection and cell graphs

- **Windows** (`LCV/cv_image_parts_20261110.py:35-43`; same code at `MC/legible_window_features_20261105.py:37-43` and `CRS/deploy_hexkonan.py:22-29`): count peaks >= 0.5 on a 16 px grid, score every 380x300 window at stride 16 by its count, take the top 3 greedily with the neighbourhood of each pick (half a window each way, in grid cells) suppressed.
- **Graphs** (`LCV/cv_image_parts_20261110.py:46-79`; HEX twin `LHX/graph_prep_lane.py:13-46`): nodes = peaks >= 0.35 inside a window; an image gets graphs only with >= 10 peaks >= 0.5 and >= 16 nodes, and a window only with >= 12 nodes. Edges = Voronoi ridge pairs. Nine node features: confidence; log(cell area / median area of bounded cells) clipped to [-2, 2]; log(mean neighbour distance / median) clipped; log(nearest-neighbour distance / median) clipped; side count clipped to [3, 9]; mean neighbour confidence; unbounded flag; x and y relative to the window. At packing, sides become (s - 6) / 2 and positions 2p - 1 (`:82-86`).
- **Graph network** (`LCV/cv_image_parts_20261110.py:90-117`): d 64, 5 message-passing layers, mean and SD pooling over all nodes of the image's windows, 4 zero context inputs, head to 3 outputs (log residual of CD, CV, HEX against H; CD is masked out upstream). An image with no graph gets the network's empty-graph output (`:102-105`). Output = mean over all checkpoints of the pattern. These nets run on CPU in float32.

## 2. CD flow (tier p = CD t1)

Entry: `INF/recipe_infer.py:232-239` -> `LCD/cd_combine_20261110.py:14-20`. Parameters: `PP.cd` (= `CDP.deploy`).

1. **P1 CD.** Section 1.2 P1 value x SCALE_CD 0.9970446 (`INF/recipe_infer.py:67`), then x `p1_cd_multiplier` 1.000382888507452 (`INF/recipe_infer.py:238`). Effective scalar on the raw member mean: `P1_scalar_on_raw` 0.9974263680018339.
2. **A CD.** Section 1.2 A value x 0.9914125837684035, then `/ A_scalars[0] * recipeA_cd_scalar` (`INF/recipe_infer.py:237`). In tier p both numbers are 0.9914125837684035, so the second factor is 1.
3. **D** and **V** as in section 1.2, no scalar.
4. **Log-space blend.** `lq = 0.25 log P1 + 0.40 log A + 0.10 log D + 0.25 log V` (`LCD/cd_combine_20261110.py:15`; `PP.cd.weights`).
5. **Mask-density slot.** If `cd_mm2` is finite and > 0 and `n_int >= 50` (`mask_min_cells`): `lq <- lq + 0.1 * (2.0601792021173932 + 0.7457325382234986 * log cd_mm2 - lq)` (`LCD/cd_combine_20261110.py:16-18`; `mask_a`, `mask_b`, `mask_w`). Otherwise `lq` is unchanged.
6. **Log-affine stretch.** `CD = 0.9975417775193984 * exp(1.075 * (lq - 7.917583509846655) + 7.917583509846655)` (`LCD/cd_combine_20261110.py:19-20`; `stretch_c`, `stretch_g`, `stretch_mu`).
7. No clipping, no rounding (section 7).

How the parameters were fitted (`LCD/cd_tiers_20261112.py:43-65`, function `fit(tr, ...)`):

- Scalars: weighted median of y/p with weights p/y on the training rows (`:39-40, :44`).
- Weights: exhaustive grid, A 0.25-0.75, D and V 0-0.40, step 0.05, P1 = remainder >= 0.1 (`:47-51`); objective = training-row CD MAPE after a stretch fitted on the training `good` rows (`:53-55`). `weight_on_grid_edge` is empty for the deploy fit and every fold (`CDP`).
- Mask slot: `mask_a`, `mask_b` by least squares of log label on log `cd_mm2` over training rows with `n_int >= 50`, label > 0, `good` (`:56`); weight from {0, 0.05, 0.10, 0.15, 0.20} by the same objective (`:57-59`). The 50-cell threshold is fixed (`:62`).
- Stretch: section 5.
- Strict value: the whole `fit` on the four other folds per scored fold (`:79-80`). Deploy: the same function with every row as training (`:86`). L5: deploy parameters through `combine()` reproduce the in-sample search to 3.7e-15, per-fold parameters reproduce the strict OOF to 3.8e-15 (`CDP.L5`).
- Per-fold strict fits of t1 (`CDP.per_fold`): weights 0.25/0.40/0.10/0.25 on folds 0-2 and 0.25/0.45/0.10/0.20 on folds 3-4; `mask_w` 0.1 and `stretch_g` 1.075 on all five; `mask_a` 2.009-2.156, `mask_b` 0.7335-0.7523; `stretch_c` 0.99727-0.99793.

Measured effects (CD MAPE, five folds):

| Step | Value | Baseline and folds | Source |
|---|---|---|---|
| Whole CD stack, tier p | 6.17366 strict | Best pipeline 6.2610 strict / 6.2639 as computed, same five folds: -0.087 | `CDP` (`strict_five_fold_cd`); [O-CD-TIERS] |
| Tier p against the full freeze (D-CD-001) | -0.0006 (donor-bootstrap CI -0.007 to +0.0057); per fold +0.0015 / +0.0111 / -0.0021 / -0.0156 / +0.0020 | Incumbent 6.17428 strict recomputed in the same run | `CDP` |
| Log-space (geometric) P1 / recipe A reweighting | -0.035 | Hybrid B, cross-fit (11-08) | `final_summary_cd_20261113.md` 4a |
| DINOv3-B member | -0.047 | Five folds (11-08) | same |
| ViT-L member | Stack 6.1775, -0.086 against the best pipeline (11-09). t2 (t1 without ViT-L) 6.1875: +0.013 against the incumbent 6.1743 | Five folds, strict for the tiers | same; [O-CD-TIERS] |
| D and V together | t3 (P1 + recipe A + mask only) 6.2138: +0.040 against the incumbent (CI +0.017 to +0.062) | Incumbent 6.1743 strict | [O-CD-TIERS] |
| Mask-density slot | -0.017, all five folds lower (11-09) | CD stack without the slot | `final_summary_cd_20261113.md` 4a |
| Stretch on CD | Hybrid B CD 6.312 -> 6.264 | Nested five folds, hybrid B | `meta_ceiling_20261103/cpu/stretch_nested.json` (`hex6_nested` vs `stretch_B`) |
| Strict nesting of the whole CD procedure | 6.1741 as computed, 6.1743 strict (+0.0002) | D-CD-001, five folds | [D-CD-001] |

## 3. CV flow (tier p = CV deploy_t1)

Entry: `INF/recipe_infer.py:245-252` -> `LCV/cv_apply_20261110.py:67-90`. Parameters: `CVP`.

1. **Base.** `A_cv` = recipe A CV x 0.9891082957404466; `P1_cv` = P1 CV x 0.987972130802936; `H_cv = 0.5 A_cv + 0.5 P1_cv` (arithmetic; `INF/recipe_infer.py:249`).
2. **GRAPH term** `graph_cv`. Single-view graphs of s46 / s47 / s48 go to the sets noctx / pre47 / pre48 (`INF/recipe_infer.py:107`); per set the mean over its nets (10 / 3 / 3 seeds x epoch snapshots 21-25 = 80 files), then the mean over the 3 sets; CV column (`INF/recipe_infer.py:112`; `CVP.graph_ensemble`).
3. **Simulation-embedding ridge** `ridge_cv`. Three `sim2_l8` graph encoders give a 128-d pooled embedding each (mean over the graph sets that have a graph; NaN if none; `LCV/cv_image_parts_20261110.py:134-146`). Per encoder seed: `((e - mu) / sd) . w + t0`, zero when the embedding is missing; mean over the 3 seeds (`LCV/cv_apply_20261110.py:57-64`). Ridge strength 30000 for CV, 3000 for HEX; `t0_CV` 0.0017076599704892556, `t0_HEX` -0.004238582046492172 (`cv_ridge_params.npz`; `cv_ridge_lambda.json`).
4. **Plain and order-loss graph quartets** `qplain_cv`, `qorder_cv`. Four graph sets (TTA peaks of s46, s47, s48, and of the mean map), nodes sorted by confidence; per set 3 seeds x epochs 21-25; mean over nets, then over the 4 sets (60 files each; `INF/recipe_infer.py:108, 113-114`).
5. **DINOv3 term** `dino_logcv` (section 1.2).
6. **Frozen token-statistic heads.** Patch tokens (124x164, 768) of the recipe A s46 epoch 20 encoder, one unflipped view (`INF/members.py:188-201`). Four blocks (`LCV/cv_image_parts_20261110.py:161-179`): token SD (768); radial power spectrum of the 64 PCA components in 24 log-spaced bands (1536); symmetric square root of the 64x64 PCA covariance, upper triangle (2080); autocorrelation at lags 1, 2, 3, 4, 5, 6, 8, 10 (6144). PCA basis from `lane_gpu2_20261106/frozen_s46/pca.npz`. Per block 3 seeds x epochs 16-20 = 15 heads; log CV = output x sigma + mu; mean of the 15 (`:190-208`).
7. **Six-component stack** (`LCV/cv_apply_20261110.py:67-82`; `CVP.weights`):
   `L = log H_cv + graph_cv + ridge_cv + 1.0 (qplain_cv - graph_cv) + 0.25 (qorder_cv - graph_cv) + 0.125 (dino_logcv - log A_cv) + 0.25 (fz_spectrum - fz_std) + 0.375 (fz_cov_sqrt - fz_std) + 0.25 (fz_autocorr - fz_std)`.
   With the weight 1.0 the plain quartet replaces the GRAPH term; `graph_cv` keeps a net coefficient of -0.25.
8. **Stretch.** `CV = 0.9806061830810574 * exp(0.975 * (L + 0.9812067899179452) - 0.9812067899179452)` (`LCV/cv_apply_20261110.py:85-90`; `CVP.stretch_cv` = [g, c, mu]).
9. No clipping, no rounding.

How the parameters were fitted:

- Weights: coordinate descent over {0, 0.125, ..., 1.0}, two passes (`LCV/cv_tiers_20261112.py:86-96`). Strict: for scored fold f the objective is the inner four-fold MAPE with each inner fold stretched on folds outside {f, g} (`:71-76, :100-101`). Deploy: the same descent with no outer fold (`score_all`, `:79-83, :102`). L5: both objectives on all rows give the deploy weights; stretch max abs difference <= 1.1e-16 (`deploy_t1/README_tier.json`, key `l5`).
- Ridge: strength by leave-one-fold-out over the five folds on the grid 300 ... 100000 (`LCV/cv_deploy_fit_20261110.py:48-61`; MAPE by strength in `cv_ridge_lambda.json`: CV 10.2768 at 30000, HEX 9.8187 at 3000). Weights fitted on all `good` rows against `log y - log(H exp(GRAPH_full))`, where GRAPH_full is the all-rows graph nets' in-sample prediction (`:72-87`). With an outer fold the recipe reproduces the nested BEST - GRAPH to <= 3.6e-9; the per-fold strengths were 1e5 / 3e4 / 1e4 / 3e4 / 1e5 [O-CV-L5-001].
- Stretch: on all `good` rows of the deploy pre-stretch prediction, with the graph parts in-sample from the all-rows nets and the DINOv3 and frozen-head terms out-of-fold (`LCV/cv_deploy_fit_20261110.py:9-10, 94-95`). The same stretch fitted on the OOF pre-stretch values is stored for comparison only: [0.975, 0.9786712908405326, -0.9790259836955335] (`CVP.stretch_cv_fit_on_oof_for_comparison`).
- Scalars: recipe A on all rows (`:74-78`); Phase-1 is the fixed SCALE (`:103`).

Measured effects (CV MAPE, five folds):

| Step | Value | Baseline | Source |
|---|---|---|---|
| Whole CV stack, tier p | 10.1115 strict | Best pipeline 10.1512 as computed (10.14786 with scalars refit) | `README_tier.json`; `docs/BRIEF.md:77-78` |
| Tier p against the freeze (three DINOv3 seeds) | -0.0045; per fold +0.0019 / -0.0362 / +0.0053 / +0.0016 / +0.0051 | Incumbent 10.1160 strict, same run | [O-CV-TIERS-001] |
| Frozen heads | Dropping them costs +0.0224 | As computed, D-CV-001 stack | `final_summary_cv_20261113.md` 4a |
| Both graph quartets | Dropping them costs +0.0118 strict (10.1115 -> 10.1233) | Tier t1, same run | [O-CV-TIERS-001] |
| Order-loss quartet | 10.1101 with, 10.1143 without | As computed | `final_summary_cv_20261113.md` 4a |
| DINOv3 term | No DINOv3 term: 10.1238 against 10.1160 (+0.0078) | Strict, same run | [O-CV-TIERS-001] |
| Graph term and simulation-embedding ridge (best pipeline) | Mean of the three metrics 8.807 -> 8.751 (graph) -> 8.739 (ridge) | OOF, best-pipeline ladder | `MC/best_pipeline_20261106.py:2` |
| Strict nesting (inner stretch) | 10.1101 as computed, 10.1160 strict (+0.006) | D-CV-001 | lesson L4; [D-CV-001] |

## 4. HEX flow (tier p = HEX t2, 38 columns, then the cell-route step)

Entry: `INF/recipe_infer.py:159-189` -> `LHX/hex_apply.py:61-68` -> `LHX/joint_deploy.py:45-75` -> `CRS/deploy_hexkonan.py:52-64`. Parameters: `PP.hex.P` (= `HXP`), `CRP`.

### 4.1 Inputs built by non-network algorithms

1. **A, P1.** HEX columns with their scalars (0.9915938787867987 and 0.9892234872488541).
2. **hex6 feature f6.** Ten detectors (`H6P.families`: three stride-8, one multi-layer stride-8, five stride-2 specialists, pre-only s46), one view each. Peaks at 0.35 (`H6P.threshold`): stride-8 maps by 3x3 local maxima with centroid refinement (`GT/deploy/hex6_correction.py:43-53`), stride-2 maps by `peaks2`. Per detector `hex6_all` = share of 6-sided Voronoi cells among bounded cells whose vertices lie inside the Delaunay hull, + 1e-3; NaN with < 20 peaks or < 12 bounded cells (`:56-69`). f6 = mean over detectors of log hex6; `have` = at least one detector has a value (`:124-127`; `INF/recipe_infer.py:122-127`).
   - Tier p does not apply the standalone hex6 correction `exp((1-w) log HEX + w (a + b f6))`. It uses f6 inside the ridge base and as ridge columns (section 4.2), with the constants hard-coded as -0.2084, 0.5989 and 0.28 / 0.72 (`LHX/joint_deploy.py:49-50`). `H6P` holds a = -0.208355602442686, b = 0.5989379720468397, w = 0.28, fitted on all 8,925 usable rows (`H6P.rows_fit`; `GT/deploy/fit_blend.py:60-66`: a, b by least squares of log HEX on the feature, w by MAPE on a grid). Only the family list of `H6P` is read at inference.
   - Measured effect of hex6 as fitted: leave-one-fold-out -0.0934 HEX points on hybrid B, five of five folds improving, per-fold w 0.26 / 0.28 / 0.28 / 0.28 / 0.32 (`H6P.lofo`); nested read HEX 10.115 -> 10.022 (`stretch_nested.json`, `hex6_nested`).
3. **best, graph.** `graph` = HEX column of the GRAPH term (section 3 step 2); `best = graph + ridge_HEX` (section 3 step 3) (`INF/recipe_infer.py:166`).
4. **g_s10x3.** Mean of the HEX outputs of the CV lane's `gfull_noctx` nets and the HEX lane's `full_d64_l5_pre47_s10` and `full_d64_l5_pre48_s10` nets (`INF/recipe_infer.py:171-172`). **g_hex46** = HEX output of the HEX-only graph net `full_hex_pre46_s5` (5 seeds x 5 epochs) (`:173`). HEX-lane nets: mean over all saved seeds and epochs (`LHX/gnn_hex_infer.py:41-57`).
5. **Image statistics** (section 1.1), entering as log(mean+1), log(SD+1), black share, log(Laplacian variance+1) (`LHX/joint_deploy.py:57`).
6. **Count-matched prior `cm1.0_h6c`** (`LHX/feat_prior.py:33-64`). Expected cell area from the submission-1 hybrid CD, `cd_hybB = 0.5 A_cd + 0.5 P1_cd` (`INF/recipe_infer.py:158`): `A = 1e6 / CD / 0.7716049^2` px, spacing `d = sqrt(2A / sqrt 3)`. Peaks >= 0.2 of the 3 pre-only detectors (single view) are linked within 0.4 d, scored by the summed best confidence per detector / 3, and thinned by NMS at 0.55 d. In each of the 3 windows the top `1.0 * 380*300 / A` candidates are kept; the feature is the mean over windows of the 6-sided share among "clean" cells (area within 0.55-1.6 of the median and no implausible neighbour; `LHX/feat_fuse.py:50-66`).
7. **Ridge evidence k3** (5 values; `LHX/feat_ridge.py:24-57`), on the s46 single-view graph: per window, Voronoi ridges are measured for length relative to the window median and for darkness (1 - mean image along the ridge / mean image at the two centres). Side counts keep ridges longer than 0.15 / 0.25 / 0.35 of the median, or darker than half the median darkness; features = 6-sided share of interior cells under each count, plus mean darkness, averaged over windows by cell count.
8. **16-detector consensus `ridge_fall16_h6_s15`.** Peaks >= 0.2 of the 16 detectors linked within 0.4 x spacing (spacing = median nearest-neighbour distance of the densest detector at >= 0.35), score = sum of best confidences / 16, kept at score >= 0.1 (`LHX/fuse_peaks.py:19-30`); graph nodes at score >= 0.25 (`LHX/hex_test_features.py:103-106`); feature = the 0.15 ridge-evidence share on that graph (`:110`).
9. **Segmentation block.** The 17 mask features of section 1.4.
10. **cvw3** and **attnp5** member values (section 1.2). The attention-pool network: DINOv3-B tokens, LayerNorm, multi-head attention with 4 learned queries and 8 heads, the 4 outputs concatenated, LayerNorm(4d), Linear(4d,128), GELU, Linear(128,3) (`lane_backbone_20261108/source/bb_runner_attnpool_20261111.py:102-115`; `INF/members.py:145-159`).

### 4.2 The joint ridge (38 columns)

Code: `LHX/joint_deploy.py:45-75`. Parameters: `HXP.deploy`.

1. **Base.** `bh = exp(0.7 log P1 + 0.3 log A)` (`w_ap1` 0.7). Where `have`: `hh = exp(0.72 log bh + 0.28 h6)` with `h6 = -0.2084 + 0.5989 f6`; else `hh = bh`. `base = hh * exp(best)` (`:48-51`).
2. **Design matrix** in block order (`:52-66`), standardised weights from `HXP.deploy.w` in the same order:

| Block (columns) | Content | Weights |
|---|---|---|
| blend (4) | log P1, log A, h6 (0 without `have`), `have` | -0.26505, -0.11451, -0.14922, -0.04148 |
| best_graph (2) | best, graph | -0.14015, 0.03538 |
| graphens (1) | g_s10x3 | -0.04605 |
| gnn46 (1) | g_hex46 | 0.00143 |
| imgstats (4) | section 4.1 item 5 | 0.00369, -0.00418, 0.00048, -0.00365 |
| prior (1) | log cm1.0_h6c | 0.00049 |
| ridge_k3 (5) | log of the four shares, darkness | 0.01183, -0.00814, -0.00477, -0.00622, 0.00192 |
| ridge_fall16 (1) | log h6_s15 | 0.00390 |
| segm (16) | log(h6, h5, h7 + 1e-3), log(n_int+1), log area_cv, log cd_mm2, coverage, 7 side-count shares (3..9), log(h6_strict + 1e-3), usable flag | -0.05679, 0.03857, -0.00066, -0.03934, 0.02440, 0.05750, 0.01669, -0.00246, -0.01036, -0.04406, 0.05146, -0.01052, -0.00425, 0.00165, 0.02821, 0.01135 |
| cvw3_mean (1) | log cvw3 HEX | -0.00254 |
| attnp5 (1) | log attention-pool HEX | 0.01884 |
| log base (1) | log base | 0.55551 |

   - Segmentation block (`:31-34`): usable only if the first 15 features are finite and `n_int >= 50`; otherwise the 15 values are 0 and the flag is 0. A missing `h6_strict` is filled with 0.5510068833827972 (`h6_strict_fill`, the median over usable rows, label-free; `:111`).
3. **Imputation and ridge.** Non-finite entries take the training median (`HXP.deploy.median`). `lc = log base + ((X - mean) / sd) . w - 0.0003804250946470359` (`:73-74`; `intercept`). Ridge strength 1.0 (`alpha`).
4. **Stretch.** `HEX_ridge = 0.9952149391615938 * exp(1.025 * (lc + 0.6180044007826191) - 0.6180044007826191)` (`:75`; `HXP.stretch`).

How the parameters were fitted (`LHX/joint_deploy.py:83-104`, function `outer(f, ...)`; f = None is the deploy fit):

- Target `log y - log base`; ridge strength from {1, 10, 100, 1e3, 1e4, 1e5} (`:12`) by leave-one-fold-out over the available folds (`:94-100`); base kind `ap1` against the best-pipeline base by the same inner error (`:102`; `HXP.base_inner_errors`: deploy ap1 0.095405, hb 0.095508); fit on the available `good` labelled rows; stretch on them (`:103`).
- Nested picks: strength 10 / 1 / 1 / 1 / 1 and `w_ap1` 0.7 / 0.7 / 0.7 / 0.7 / 0.8 for folds 0-4; stretch g 1.025 on folds 0, 1, 2, 4 and 1.0 on fold 3 (`HXP.nested`, `HXP.nested_stretch`).
- Deploy: strength 1.0; `w_ap1` is set to 0.7 for the no-outer-fold call (`:89`: `w = wf[f] if f is not None else 0.7`); base on every row = the CV lane's all-rows BEST in-sample (G + RID from `lane_cv_20261108/deploy/cv_deploy_insample_members.npz`, `:113`), the "chain convention".
- L5: rerunning `outer(None)` reproduces the stored deploy parameters, max difference 0.0 (verifier <= 1.5e-13) [A-HEX-006, O-VERIFY-HEX-TIERS].

Measured effects (HEX MAPE, five folds, chain convention):

| Step | Value | Baseline | Source |
|---|---|---|---|
| 38-column ridge of tier p (before the cell-route step) | 9.72093 strict (nested 9.71991; folds 0+4 9.84918) | Best pipeline 9.80262 strict: -0.0817 | `HXP.oof_hex_five_folds`; [O-HEX-010]; `corr_hex_e2e_strict_t2.json` |
| Joint ridge instead of the 10-step chain | 9.72662 (38 columns at the time) against 9.74961 chain procedure | Strict | [O-HEX-001] |
| CPSAM multi-target column (absent in tier p) | Removing it costs +0.00116 (9.71977 -> 9.72093) | 39-column ridge, strict, same run | [O-HEX-010] |
| Attention-pool column | Removing it from t2 costs +0.0080 (9.72093 -> 9.72891); 39 against 38 columns -0.0069 | Strict, same run | [O-HEX-010], [O-HEX-001] |
| cvw3 column | Removing it next costs +0.0031 (9.72891 -> 9.73200) | Strict ladder | [O-HEX-010] |
| 16-detector consensus column | Removing it next: -0.0002 (9.73181) | Strict ladder | [O-HEX-010] |
| HEX-only graph net column | Removing it next: +0.0009 (9.73270) | Strict ladder | [O-HEX-010] |
| Zero-shot segmentation block | Ridge without it 9.75186 nested | 39-column ridge nested 9.71882 | `final_summary_hex_20261113.md` 4a |
| Stretch after hex6 (best pipeline) | HEX 10.022 -> 9.958 | Nested, hybrid B + hex6 | `stretch_nested.json` (`hex6_then_stretch_vs_hex6`) |
| Strict nesting of the ridge | 9.71991 nested, 9.72093 strict (+0.0010) | Tier t2 | above |

### 4.3 Cell-route step ("device rule from detections", kept version "P2 win+full")

Code: `CRS/deploy_hexkonan.py:32-64`, `CRS/cr_lib.py:142-183`. Parameters: `CRP`. Called at `INF/recipe_infer.py:188-189`.

The "device rule" is the Konan centre-method weighted-bisector polygon rule (`kg/wiki/methods/method-konan-weighted-bisectors.md`): cell polygons are built from cell centres, and CV and HEX are read from the polygons.

1. **Frame spacing s.** Median nearest-neighbour distance of the mean-map TTA peaks ("pk3") with confidence >= 0.5; if 20 or fewer such peaks, of those >= 0.35. No features if pk3 has < 20 peaks or the reference set < 10 (`CRS/deploy_hexkonan.py:34-37`).
2. **Point set P2.** Sources: TTA peaks of s46, s47, s48 with confidence >= 0.35, and mask centroids with area >= 0.15 s^2 (given confidence 0.5). Greedy cross-source vote: the highest-confidence point takes the nearest unassigned point of each other source within 0.45 s; cluster mean; clusters with >= 1 source are kept, i.e. a union with cross-source merging (`:38-40`; `CRS/cr_lib.py:169-183`).
3. **Cells.** Delaunay over the whole frame; a seed is used if it is not on the hull, has >= 3 neighbours and no angular gap between neighbours above 100 degrees. For exponent k in {0, 2}: the cut point on each edge i-j divides it as `d_i^k : d_j^k` (d = mean neighbour distance; k = 0 is Voronoi), the polygon is clipped from a square of half-width 3 d_i; area by the shoelace formula; sides = edges longer than 0.08 d_i (`CRS/cr_lib.py:142-166`).
4. **Regions and statistics.** "win" = seeds inside the 3 windows of section 1.6 (computed on the spacing reference set), "full" = all interior seeds. Per region: `cv_k0`, `cv_k2` (SD / mean of areas), `hex_k0`, `hex_k2` (share of 6-sided cells), `n`; NaN statistics when a region has < 10 cells (`CRS/deploy_hexkonan.py:41-49`). Ten features, names in `CRP.names`.
5. **Ridge correction.** Design: log of the two CVs (floored at 1e-4), the two HEX shares, log1p(n), for win then full, plus log of the ridge HEX (11 columns). A row is usable if all are finite and the full-frame cell count >= 50 (`:52-58`). Usable rows: `HEX = HEX_ridge * exp(((X - mu) / sd) . w) * c / c0`; other rows keep `HEX_ridge` (`:61-64`).
   - Shipped values (`CRP`): w = [0.0010091, -0.0010018, 0.0021478, -0.0021090, 0.0002080, 0.0023443, 0.0006513, 0.0068729, 0.0033972, 0.0021188, -0.0116213]; c = 1.0001677811343108, c0 = 1.000454458839663 (ratio 0.9997135); ridge strength 100; `train_rows` 8825; base `chain_joint_t2.npz final_hex`.
6. No clipping, no rounding.

How the parameters were fitted (`INF/params_build.py:64-68, 145-146`; same recipe as `CRS/corr_newbase.py:13-16`): target `log y - log HEX_ridge` on the usable, `good`, labelled rows; ridge with fixed strength 100; `c` and `c0` = weighted medians of y/q and y/base (weights q/y, base/y). Tier p: fitted on all rows against the tier-t2 nested OOF HEX. Of the 8,954 rows, 8,845 are usable and 8,825 are training rows (recomputed in this analysis).

- Fixed in advance and not tuned, per `CRS/corr_strict.py:2-4`: strength 100, the 50-cell rule, the feature-construction thresholds. The point-set versions (P0-P3) were chosen on the 25 expert-marker frames (`CRS/feats_all.py:2-3`); the exponent k = 2 was fitted on the same 25 frames (`method-konan-weighted-bisectors.md`, literature note).
- Not nested in the kept step: the choice of the feature version among 15 versions and "no step", made on five-fold all-label reads (`CRS/corr_strict.py:3-5`). The strict procedure nests that choice by inner four-fold CV (`:50-60`).
- L5: for the full / t1 file the no-outer-fold fit reproduces the parameters exactly and the no-outer-fold version choice picks the kept version (gap to the second 0.00207) [O-DEPLOY-002, O-DEPLOY-008, F-VER-002]. For `cellroute_deploy_p.json` the records hold no such check (`final_summary_deploy_20261113.md` section 6). In this analysis the kept-version fit with no outer fold on `chain_joint_t2.npz` was recomputed read-only: maximum absolute difference to `CRP` 2.8e-17 over mu, sd, w, c, c0; 8825 training rows. The no-outer-fold version choice on the t2 base was not rerun.

Measured effects (HEX MAPE, five folds):

| Read | Value | Baseline | Source |
|---|---|---|---|
| Tier p, as computed (kept version, weights nested) | 9.70635 (-0.0136) | t2 nested OOF 9.71991 | [O-DEPLOY-014] |
| Tier p, strict for the step's own choices | 9.71688 (-0.0030) | same base | [O-DEPLOY-014] |
| Tier p, end-to-end strict, chain convention | 9.71943 (-0.0015); folds 0+4 +0.0011; per fold -0.0038 / -0.0154 / +0.0020 / +0.0037 / +0.0060; without the 25 gate frames +0.0008 | t2 strict 9.72093, same run | `lane_cellroute_20261109/out/corr_hex_e2e_strict_t2.json`; [O-DEPLOY-014] |
| Tier p, end-to-end strict, OOF-base convention | 9.71785 (-0.0031); folds 0+4 +0.0053 | same | same file |
| Version picked per fold in the tier-p strict run (chain convention) | P2 win+full+creg / kept / P3 win+full / P1 win+full / kept | - | same file, `picks` |
| Full tier (39 columns), for reference | As computed 9.70581 (-0.01301); strict on the same base 9.71576 (-0.00306); end-to-end strict 9.71342 (-0.00635), folds 0+4 +0.00132 | 9.71882 nested / 9.71977 strict | [O-DEPLOY-006, O-DEPLOY-007] |

## 5. The MAPE-optimal log-affine stretch (all three metrics)

Form: `out = c * exp(g * (log p - mu) + mu)`.

How c and g are found (`MC/stretch_nested_20261103.py:11-21`; CD twin `LCD/cd_common.py:22-27`; CV twin `LCV/cv_tiers_20261112.py:59-64`):

1. `mu` = mean of log p over the fit rows with label > 0.
2. For each slope g on the grid 0.85, 0.875, ..., 1.35 (21 values, step 0.025): `q = exp(g (log p - mu) + mu)`; `c` = weighted median of y/q with weights q/y. This c minimises the fit-row MAPE of c q exactly, because `|c q - y| / y = (q/y) |c - y/q|`.
3. Keep the g with the lowest fit-row MAPE.

Shipped values and where they were fitted:

| Metric | g | c | mu | Fit rows for deployment | Per-fold strict fits |
|---|---:|---:|---:|---|---|
| CD | 1.075 | 0.9975417775193984 | 7.917583509846655 | All `good` rows, on the blend after the mask slot (`LCD/cd_tiers_20261112.py:58, 86`) | g 1.075 on all folds |
| CV | 0.975 | 0.9806061830810574 | -0.9812067899179452 | All `good` rows; graph parts in-sample, other terms OOF (`LCV/cv_deploy_fit_20261110.py:94-95`) | not read in this analysis |
| HEX | 1.025 | 0.9952149391615938 | -0.6180044007826191 | All `good` labelled rows of the no-outer-fold candidate (`LHX/joint_deploy.py:103`) | g 1.025 / 1.025 / 1.025 / 1.0 / 1.025 |

Measured effect of the stretch alone, nested five folds (`meta_ceiling_20261103/cpu/stretch_nested.json`): on hybrid B, mean of three metrics -0.018 (CI -0.028 to -0.007; per fold -0.014 / -0.012 / -0.015 / -0.021 / -0.029), with CD 6.312 -> 6.264; on recipe A alone +0.004 (CI -0.001 to +0.010); on Phase-1 alone -0.017; after hex6, a further -0.037 on the mean (HEX 10.022 -> 9.958).

## 6. Runtime guards, fallbacks and start-up checks

Entry: `INF/docker/gc_entry.py`.

**Loud failures at start-up (the job fails, nothing is written):**

| Check | Line |
|---|---|
| CUDA not available | `:38` |
| `inputs.json` does not hold exactly the one socket `corneal-specular-microscopy-image` | `:39-40` |
| Not exactly one `.mha` in the input folder | `:41-42` |
| Any of the three output files already exists | `:43-44` |
| `cellpose.vit` lacks `dinov3_vitb16` (the CPDINO backbones did not import) | `:47-48` |
| A file of the manifest lists `p1_members`, `cvw3`, `cps5` is missing (only these three lists are checked) | `:51-52` |
| CV parameter folder missing (`Recipe.cvp is None`) | `:53-55` |

Build-time checks: torch 2.11.0, timm 1.0.20, transformers 5.3.0 asserted; tier manifest and parameters must exist; every code file readable by uid 10001 (`INF/docker/Dockerfile.freeze:23, 33-37, 41-44`).

**Per-image guard (`:57-65`).** One `try` around read, conversion and prediction. Any exception gives the three fallback constants for all three outputs; there is no partial fallback per metric or per member. Cases that reach it:

- Unreadable MHA (SimpleITK raises).
- Frame shape other than (972, 1296) after conversion (`:59`).
- Any member, detector or builder raising (including a missing weight file outside the three checked lists).
- Any output not finite or not > 0 (`:62`).
- Any stubbed input in deploy mode (`:63`).

**Input conversion `coerce` (`:23-32`), applied before the shape check:** leading singleton axes are dropped; a last axis of 3 or 4 is averaged over its first 3 channels (RGB to grey); any other 3-D array keeps its first plane; a non-uint8 array is min-max scaled to 0-255 with rounding. The notes go to the job log.

**Fallback constants:** CD 2770, CV 0.36, HEX 0.53 (`:19`). Origin: the MAPE-optimal constants of submission 1, the weighted medians of the 8,954 audited training labels with weights 1/y (`submission_hybrid_20261030/hybrid_guard_20261031.py:9-10`, local repo; `results/submission1_package_20261031/PACKAGE.md:16`). The script that computed them: not found.

**Output writing (`:66-68`):** `json.dump(float(v), allow_nan=False)` into files opened in exclusive-create mode. The job log line carries mode (`recipe` or `constant`), errors, notes, hook statistics and the decisions list (`:70-71`).

**Fallbacks inside the recipe (no exception, the prediction continues):**

| Situation | Behaviour | Code |
|---|---|---|
| Mask has < 10 labels or < 5 interior cells, or `n_int < 50` | CD keeps the member blend without the mask slot | `INF/seg_run_feats.py:11, 24`; `LCD/cd_combine_20261110.py:16-18` |
| Same, HEX | Segmentation block = zeros with flag 0 | `LHX/joint_deploy.py:31-34` |
| No detector yields hex6 | f6 = 0, `have` = False: base without the hex6 blend, h6 column 0 | `INF/recipe_infer.py:127`; `LHX/joint_deploy.py:49-50` |
| Any non-finite HEX ridge column (prior, ridge evidence, consensus) | Training median | `LHX/joint_deploy.py:73` |
| Image without a graph | Graph nets return their empty-graph output; simulation-embedding ridge returns 0 | `LCV/cv_image_parts_20261110.py:102-105`; `LCV/cv_apply_20261110.py:61-62` |
| Cell-route features missing or < 50 full-frame cells | HEX keeps the ridge value | `CRS/deploy_hexkonan.py:57, 64` |

Measured behaviour of the guard on the built image (build box, uid 10001, 17 of 17 jobs exit 0): a garbage file gives the constants in 6 s; all real frames run in recipe mode [O-DEPLOY-022]. Packaging faults the guard turned into constants before they were fixed: missing PYTHONPATH, constants on 13 of 13 real-T4 jobs [F-DEPLOY-002]; a mode-600 code file, constants on 9 of 9 jobs of the first p image [F-DEPLOY-005]. Two further faults were found on the real T4: a filtered timm file (ModuleNotFoundError at ViT-L) and unloading not set in the image (CUDA out of memory at the attention-pool stage after 355 s) [F-DEPLOY-003].

## 7. Clipping, range limits and rounding

- **Final outputs:** no clipping, no range limit and no rounding anywhere between the stacks and the JSON files (`LCD/cd_combine_20261110.py:20`, `LCV/cv_apply_20261110.py:85-90`, `LHX/joint_deploy.py:75`, `CRS/deploy_hexkonan.py:63-64`, `INF/docker/gc_entry.py:66-68`). The only range test is "finite and > 0", which switches to the constants.
- Clipping of the final CD was read as a post-hoc lever by the CD lane and closed at +0.038 [F-CDANAT-001] (`final_summary_cd_20261113.md` section 4b).
- **Internal clips:** image scaling to [0, 1]; graph node features to [-2, 2]; side counts to [3, 9]; cell-route log features floored at 1e-4; covariance eigenvalues floored at 0 before the square root (`LCV/cv_image_parts_20261110.py:64, 179`; `CRS/deploy_hexkonan.py:54`).

## 8. Numerics that act like post-processing

| Item | What the code does | Measured output deviation | Source |
|---|---|---|---|
| fp16 storage of weights | For the paths in `INF/docker/fp16_files.txt` (82 lines; 35 of them in tier p) floating tensors with > 16 elements are stored as float16 and upcast to float32 right after `torch.load` (`INF/pack_hooks.py:23-34, 216-224`; rule at `INF/docker/convert_fp16.py:24`). In tier p: the 5 Phase-1 members, DINOv3-B s46b and s47, cvw3 s46 and s47, ViT-L s46 and s47, the attention-pool checkpoints, the 3 simulation nets. Not fp16: detectors, hex6, `cpsam_v2`, recipe A epoch 20, graph models, frozen heads | Against fp32, 20 frames, full candidate: median / max CD 0.009 / 0.027 %, CV 0.005 / 0.021 %, HEX 0.007 / 0.026 %. fp16 for every tensor: CV 0.168 %, HEX 0.381 % median (old stacks). A tier-p-specific run of this comparison: not found | [O-DEPLOY-005, O-DEPLOY-012]; `final_summary_deploy_20261113.md` section 2 |
| Base weights not shipped | Pretrained base files are replaced by zero tensors of the recorded shapes, or the model is built from its config; every member's strict checkpoint load overwrites them (`INF/pack_hooks.py:37-39, 216-244`) | Outputs bit-identical with noise in place of the base tensors (test of 2026-11-11) | `INF/pack_hooks.py:10-11` |
| Model unloading | `CLEAR_EC_UNLOAD=1` drops every cached model after each stage (`INF/recipe_infer.py:199-203`; set in `gc_entry.py:14` and `Dockerfile.freeze:15`) | GPU peak 17.35 -> 4.33 GB (full tier); outputs against the run without unloading: CD identical 20/20, CV <= 8e-9, HEX <= 9e-9. Tier p GC-style jobs against in-process: CD identical 20/20, CV <= 7.3e-9, HEX <= 1.3e-8 | [O-DEPLOY-015, O-DEPLOY-017] |
| Autocast choice | Members use bf16 where supported, else fp16 (`INF/members.py:16`). Below compute capability 8.0 (the T4), or with `CLEAR_EC_AUTOCAST=fp16`, `pack_hooks` reports bf16 as unsupported and maps every CUDA bf16 autocast to fp16, including modules that hard-code bf16 (detectors, hex6, attention pool, Phase-1); Cellpose-SAM then runs float32 (`INF/pack_hooks.py:84-114`; `INF/recipe_infer.py:53`) | fp16 + 1 detector copy against the reference (fp32 weights, bf16 compute), tier p, 200 labelled rows: median / max CD 0.035 / 0.21 %, CV 0.175 / 0.83 %, HEX 0.45 / 3.86 %. Strict-score impact applied to the t1 strict OOF: +0.0067 (95 % +0.0017 to +0.0122), HEX +0.018 (+0.004 to +0.034). Directly measured MAPE change on the 200 rows: -0.006 (-0.039 to +0.029) | [O-DEPLOY-021] |
| Same, tier t1 | - | 200 rows: 0.035 / 0.21, 0.175 / 0.83, 0.44 / 3.9 (HEX p90 1.06); impact +0.0066 (+0.0016 to +0.0121). 20 frames: +0.0065. HEX carries about 3/4 of every impact | [O-DEPLOY-020] |
| All-fp32 path | Every autocast region fp32, Cellpose-SAM float32, TF32 off, 1 detector copy | Tier p, same 200 rows: impact +0.0069 (+0.0017 to +0.0125) against fp16's +0.0066 (+0.0011 to +0.0119); HEX +0.0187 against +0.0178. Paired fp32 minus fp16 on the rows: +0.0065 (-0.0074 to +0.0204). On the real T4, all-fp32 did not finish within 600 s | [O-COORD-F32-200]; [O-DEPLOY-T4-COLAB] |
| Other kernels | - | 20 frames, impact: mixed (fp32 detectors) +0.0053; bf16 emulated +0.0062; A10G proxy (native bf16, FlashAttention-2) +0.0059. No path meets the 0.05 % median rule; 0 of 420 peak sets identical even in fp32 | [O-DEPLOY-020] |
| Number of detector copies | The 13 non-pre-only detectors run their single view as a batch of 4 identical copies so that native bf16 numerics match the batch-4 training peaks (`CRS/deploy_hexkonan_image.py:37-41, 81-85`). The T4 default sets `CLEAR_EC_DET_COPIES=1` (`INF/pack_hooks.py:101-105, 153-166`). The 3 pre-only detectors always run their 4 flips as one batch | One copy instead of four: < 5e-7 relative, fp16 and fp32. A batch of 1 under bf16 does not reproduce the stored training peaks (`diag_peaks_batch.py`, 11-10) | [O-DEPLOY-020]; `CRS/deploy_hexkonan_image.py:39-41` |
| Built image, RTX 4090 | Default (bf16, 4 copies) and forced T4 path | Default against the H200 reference: CD 0.008-0.044 %, CV 0.07-0.38 %, HEX 0.24-1.21 %. T4 path against H200: CD <= 0.073 %, CV <= 0.94 %, HEX <= 1.32 %. T4 path against the real Colab T4 (2 frames): CD 0.005 / 0.003 %, CV 0.002 / 0.033 %, HEX 0.12 / 0.06 % | [O-DEPLOY-022] |

All of the score impacts above are deviations applied to OOF predictions on an H200 with emulated paths; the platform's own T4 output was not measured (`final_summary_deploy_20261113.md` section 6).

## 9. Steps whose parameters were chosen on reported rows, and steps that are not nested (lesson L4)

Lesson L4 (ledger): choosing on reported rows was worth 0.006 (CV inner stretch), 0.010 (HEX chain step selection) and 0.010 (cell-route feature-version choice).

| Step | What is not nested | Size the records give |
|---|---|---|
| Cell-route feature version (kept "P2 win+full") | Chosen among 15 versions + "no step" on five-fold all-label reads (`CRS/corr_strict.py:3-5`) | Tier p: as computed -0.0136, strict for the step's choices -0.0030, end-to-end strict -0.0015 (folds 0+4 +0.0011; without the 25 gate frames +0.0008) [O-DEPLOY-014]. Full tier: -0.01301 / -0.00306 / -0.00635 [O-DEPLOY-006, O-DEPLOY-007] |
| Cell-route point sets and exponent k = 2 | Chosen on the 25 expert-marker frames, which are among the scored rows | Without those frames the full-tier step reads -0.0073 as computed, +0.0021 strict (nested base), -0.00356 end-to-end strict [O-DEPLOY-006, O-DEPLOY-007] |
| Cell-route strength 100, 50-cell rule, feature thresholds | Fixed in advance, not tuned (`CRS/corr_strict.py:2`) | No size given |
| P1 SCALE | Fitted on all folds of the September OOF, never refit (section 1.3); used as it is by CV and HEX | Best pipeline 8.73923 as computed against 8.73627 with P1 SCALE and recipe A scalars refit per fold (`docs/BRIEF.md:77-78`) |
| hex6 map and blend (-0.2084, 0.5989, 0.28) | Fitted on all rows; shared by every HEX number, strict values included (`final_summary_hex_20261113.md` section 6) | Per-fold w 0.26-0.32; per-fold a and b within 0.007 and 0.009 of the all-row values (`H6P.lofo`); no strict-against-fixed read found |
| CV inner stretch | The as-computed stack scored inner folds with a stretch that had seen them | 10.1101 as computed against 10.1160 strict (+0.006) [D-CV-001] |
| CV component list (the "fixed six") | The list was settled on all-label reads; the strict procedure nests weights and stretch only | From-scratch strict searches: grid 10.1176, forward selection 10.1324, fixed six 10.1160 (`final_summary_cv_20261113.md` 4b) |
| HEX ridge block list | Blocks were settled on all-label reads; the strict procedure nests strength, base kind, weights and stretch | 9.71991 nested against 9.72093 strict for t2; 9.71882 against 9.71977 for the 39-column ridge |
| HEX deploy base convention | Deploy fit uses the all-rows BEST in-sample as the base ("chain convention") | 38-column ridge: 9.72662 under the chain convention, 9.74136 under an OOF-input base (lesson L5; `final_summary_hex_20261113.md` 4c). Effect on test: unmeasured |
| HEX deploy `w_ap1` | Set to 0.7 for the no-outer-fold fit (`LHX/joint_deploy.py:89`), not re-chosen there | Nested picks 0.7 on folds 0-3, 0.8 on fold 4 (`HXP.nested`) |
| CV deploy stretch | Fitted with graph parts in-sample and the other terms OOF | c 0.98061 against 0.97867 when fitted on OOF values (`CVP`) |
| CD member list | Members were settled on all-label reads; scalars, weights, mask slot and stretch are nested | 6.1741 as computed against 6.1743 strict; from-scratch searches 6.1733 / 6.1767 against 6.1741 (`final_summary_cd_20261113.md` 4b) |
| CD blend weights | Objective is the training-row MAPE, not an inner CV (`LCD/cd_tiers_20261112.py:53-55`); nested by the outer fold | Per-fold weights differ on folds 3-4 (A 0.45, V 0.20) from the deploy weights (A 0.40, V 0.25) (`CDP.per_fold`) |
| Tier choice (p) and T4 path | Compared on strict five-fold values of the same rows | Tier differences: p 8.6682, t1 8.6662, full 8.6679 [O-DEPLOY-018]; T4 path +0.0067 [O-DEPLOY-021] |
| Attention-pool column | One trained seed; the seed-mean rule (L3) was not applied (`final_summary_hex_20261113.md` 4c) | -0.0069 (39 against 38 columns), +0.0080 when removed from t2 |

## 10. Not established in this analysis

- The script and rows that produced the P1 SCALE constants (described in two repo documents; fitting code not found).
- The script that computed the fallback constants 2770 / 0.36 / 0.53 (docstring only).
- Where and on which rows the detector thresholds (0.2 / 0.35 / 0.5), the 12 px NMS, the 380x300 window size and the count of 3 windows were chosen.
- A tier-p-specific fp16-storage parity run (the recorded one is on the full candidate).
- The no-outer-fold version choice of the cell-route strict procedure on the t2 base (only the kept-version fit was recomputed here).
- The per-fold strict CV weights and stretches of tier t1 (stored in `lane_cv_20261108/out/cv_tier_t1_20261112.json`; not read).
- Any measurement on the platform's own T4.