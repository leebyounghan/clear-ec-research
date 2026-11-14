# HEX lane freeze candidate (decision D-HEX-004; written 2026-11-09 11:50 EST; updated 2026-11-11 11:40 EST: 39-column joint nested ridge with the attention-pool column (its full-data build verified 10:52 EST); 38-column ridge and 10-step chain kept as fallbacks)

## Result

Values are five-fold / folds 0+4 OOF HEX. "Strict" = the whole procedure rerun with the scored fold's labels hidden (no constant, pick, alpha, stretch or keep decision can see it, not even through the other folds' candidates; `source/strict_nest.py`). From 11-11 HEX is quoted against the strict chain value 9.74961.

| | Nested | Strict | Folds 0+4 (strict) |
|---|---:|---:|---:|
| Best pipeline (baseline; stretch per fold on the other folds) | 9.8026 | 9.80262 | 9.90790 |
| 10-step chain, procedure (historical order, keep per fold on the other folds) | 9.74255 | **9.74961** | 9.87964 |
| 10-step chain, fixed composition (decided on folds 0+4 and the seed audit) | 9.73258 | 9.74062 | 9.86637 |
| Joint nested ridge, 38 columns (fallback 1) | 9.72621 | 9.72662 (-0.02299 vs the strict chain) | 9.85123 |
| **This candidate: joint nested ridge plus the backbone attention-pool column, 39 columns (all with full-data builds)** | **9.71882** | **9.71977** (-0.02984 vs the strict chain) | **9.84826** |

Per-fold strict deltas against the strict chain procedure: joint -0.0439 / -0.0333 / -0.0071 / -0.0177 / -0.0130; with attention pool -0.0344 / -0.0613 / -0.0087 / -0.0164 / -0.0284.

## Candidate: the 39-column joint ridge (11-11 11:40 EST)

- The 38-column ridge below plus one column: log of the backbone lane's attention-pool DINOv3-B HEX (raw track, mean log over epochs 8-12, single view, no TTA). OOF: `lane_backbone_20261108/runs/attnpool_f{f}_dinov3b16_s46/raw`.
- Files: OOF `preds/chain_joint_attnp.npz` (nested 9.71882, strict 9.71977); params `preds/deploy_params_joint_attnp.json` (`JOINT_ATTNP=1 source/joint_deploy.py build`). Nested alphas 10 / 1 / 1 / 1 / 1; deploy = `outer(None)`: base ap1, alpha 1, `HEX = 0.99561 exp(1.025 (lc + 0.61800) - 0.61800)`. Checks: nested apply vs the OOF 2.2e-16; `outer(None)` vs the stored deploy params 0.0.
- Test-time input `attnp5` = exp of the mean log HEX over `lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46/raw/runs/fold99/dinov3_b16/epoch_08..11.pt` and `final.pt` (= epoch 12), with the runner's own AttnPool network and evaluate() (call and sha256 in the backbone lane's note `backbone_attnpool_f99_handoff.md`; family_infer does not build this head). Verified by the backbone lane 10:52 EST: strict loads, held-row predictions equal the stored ones, 256-frame sanity HEX r 0.9856 with the OOF members, mean log ratio -0.0087.
- Entry: `hex_apply.hex_any(P, I, f=None)` with `P = deploy_params_joint_attnp.json`; it picks the 39-column apply from the parameter count and needs `I["attnp5"]`.

## Packaging tiers (D-PKG-002, 2026-11-12 15:05 EST; freeze files untouched)

Same procedure as the candidate, fewer columns. Strict five folds, chain convention, t1 recomputed in the same run (`source/strict_ladder.py`, `runs/strict_ladder/strict_ladder.json`). Deploy parameters by `outer(None)` (`JOINT_TAG=<tier> JOINT_DROP=<blocks> source/joint_deploy.py build`); L5 check: rerunning `outer(None)` reproduces the stored deploy parameters, max difference 0.0 for every tier; nested apply vs the tier's OOF 2.2e-16. Entry unchanged: `hex_apply.hex_any(P, I, f)` (the parameter file lists its blocks).

| Removal step | Strict five folds | vs t1 | Step cost | Folds 0+4 |
|---|---:|---:|---:|---:|
| t1: 39-column ridge (D-HEX-004) | 9.71977 | 0 | | 9.84826 |
| t2: without mtcnn5 (cps5) | 9.72093 | +0.00116 | +0.0012 | 9.84918 |
| then without the attention pool | 9.72891 | +0.00914 | +0.0080 | 9.85474 |
| then without cvw3 (t3core) | 9.73200 | +0.01223 | +0.0031 | 9.85970 |
| then without the 16-detector consensus (ridge_fall16) | 9.73181 | +0.01204 | -0.0002 | 9.85735 |
| t3: then without the HEX-only graph net (hex46) | 9.73270 | +0.01293 | +0.0009 | 9.85900 |

| Tier | Params | OOF (nested) | sha256 (params) | Drops (model files, per-frame stages) |
|---|---|---:|---|---|
| t1 | `preds/deploy_params_joint_attnp.json` (freeze file) | 9.71882 | 1fbc5384...ad072c2c | nothing |
| t2 | `preds/deploy_params_joint_t2.json`, OOF `preds/chain_joint_t2.npz` | 9.71991 | a6a942f3...f16e2ba6 | CPSAM 512 multi-target (`deploy/mtcps_c512_f99`, 5 checkpoints, 6.09 GB on disk): 5 CPSAM forwards |
| t3 | `preds/deploy_params_joint_t3.json`, OOF `preds/chain_joint_t3.npz` | 9.73216 | 744b8ba7...7e81d7f8e | t2's plus: attention-pool DINOv3-B f99 (5 checkpoints, 1.77 GB): 5 forwards with its own AttnPool head; cvw3 DINOv3-B s46 + s47 f99 (10 checkpoints, 3.43 GB): 10 forwards; the 16-detector consensus path (all16 peaks -> fusion -> fall16 graph -> ridge feature): 13 of its 16 detector passes (the 3 pre-only detectors stay for prior, ridge k3 and the graphs), and the 8 lane-built detector files (`deploy/detectors/*`, `deploy/frozen_heads/*`, 1.37 GB; the green track's sg2 files stay if the hex6 feature uses them); the HEX-only graph net `full_hex_pre46_s5` (25 small files): 25 GNN forwards on the pre46 graph (the graph itself stays for GRAPH) |
| t3core (reference) | `preds/deploy_params_joint_t3core.json` | 9.73144 | 052590ac...4dab5e | t3 minus the last two drops |

Still needed in every tier: A, P1, f6/have (hex6), BEST and GRAPH (CV lane), G_s10x3 graph nets, image statistics, prior and ridge k3 (3 pre-only detectors), the zero-shot Cellpose-SAM segmentation. The cell-route step (D-HEX-005) must be refit per tier on the tier's OOF file.

## Joint nested ridge, 38 columns (fallback 1; the base procedure, 11-11)

- One ridge on `t = log HEX_label - log base`, all corrections at once (no step order), `source/joint_fit.py` (search) and `source/joint_deploy.py` (fits, apply).
- Base: `ap1` (`exp(w log P1 + (1-w) log A)`, h6 blended at 0.28 where `have`, times `exp(BEST_hex)`), w 0.7 (fold 4: 0.8) from ap1's own nested pick; the base (best pipeline vs ap1) was chosen per outer fold on the other four folds: ap1 on all five.
- Columns (38): log P1, log A, h6, have; BEST_hex, GRAPH_hex; G_s10x3; G_hex46; image statistics (4); log prior cm1.0_h6c; ridge evidence k3 (log h6_s15/s25/s35/dark, darkness); log ridge fall16 h6_s15; segmentation group bh (15, zero where not usable) + usable flag; log cvw3 seed mean (s46 + s47); log CPSAM mtcnn5; log base.
- Nested fit per outer fold f: alpha from (1, 10, 100, 1e3, 1e4, 1e5) by leave-one-fold-out on the other four folds (picks 10 / 1 / 1 / 1 / 1), fit on those folds, per-fold stretch on them. Base on the fit rows = fold f's own BEST (the chain convention, the same one every chain number uses).
- Deploy fit = `outer(None)` in `source/joint_deploy.py`: the same procedure as each nested fit, with no outer fold (coordinator condition 11-11). Base on the fit rows = the base model fitted on those rows: nested, fold f's own BEST; deploy, the all-rows BEST in-sample (CV lane `deploy/cv_deploy_insample_members.npz` G + RID, HEX column; r 0.990 with the OOF BEST). Base kind by five-fold leave-one-fold-out (ap1 9.543 vs best pipeline 9.553), alpha 1 the same way, ridge on all labelled rows, stretch on all rows: `HEX = 0.99584 exp(1.025 (lc + 0.61800) - 0.61800)`. Check: rerunning `outer(None)` reproduces the stored deploy parameters exactly (max difference 0.0); nested apply vs `chain_joint.npz` 3.3e-16. The OOF-input deploy (alpha 100) is archived as `archive/deploy_params_joint_oofdeploy_20261111.json`; its procedure reads 9.74136 nested.
- Files: OOF `preds/chain_joint.npz`; params `preds/deploy_params_joint.json` (nested fits, nested stretches, deploy fit, stretch).
- Test-time entry: `source/hex_apply.py` `hex_any(P, I, f=None)` with `P = deploy_params_joint.json` (f = None: deploy; f = k: fold k's nested fit). Nested parity against `chain_joint.npz`: 2.2e-16. The same entry with `deploy_params.json` runs the chain fallback (parity 8.9e-16).
- Inputs `I`: as for the chain (table "Test-time path" below), plus `g_hex46` = `gnn_hex_infer.predict('full_hex_pre46_s5', g_pre46.npz)[:, 2]` (built 11-10). Every column has a full-data build.
- The chain fallback's deploy (`chain_deploy.py`, `deploy_params.json`) fits on the OOF inputs, not on the convention of its own OOF number.
- Staged 39-column variant (switch only after the attention-pool full-data build is verified, cut-off 16:30 EST): `JOINT_ATTNP=1 source/joint_deploy.py build` wrote `preds/deploy_params_joint_attnp.json` and `preds/chain_joint_attnp.npz` (nested 9.71882, parity 2.2e-16; deploy = outer(None) with the same convention: alpha 1, stretch c 0.99561, g 1.025; outer(None) check 0.0). Extra input `attnp5` = exp of the mean log HEX over the f99 raw-track epochs 8-12 of the backbone lane's attention-pool DINOv3-B (single view, no TTA); `hex_any` picks the variant from the parameter count.
- Cell-route step on top (its notes, 10:30 EST): on the 38-column `chain_joint.npz`, kept-recipe refit 9.71154 (-0.0147) / folds 0+4 9.84609; strictly nested (its version choice by inner CV) 9.72169 (-0.0045) / 9.85786 (+0.0066); file `lane_cellroute_20261109/out/corr_hex_on_joint.npz`. On the 39-column `chain_joint_attnp.npz` (decision D-HEX-005, step kept, strict value quoted): 9.70581 as computed, **9.71576 strict** (-0.0031 vs 9.71977); without the 25 gate frames +0.0021 (`lane_cellroute_20261109/out/corr_hex_strict_on_joint_attnp.npz`).

## Fallback: the 10-step chain (11-11 06:45 EST)

- The lane changes the HEX column only. CD and CV are the best pipeline's.
- The chain is fixed by the seed-audit rule (section "Seed audit"), not by the per-step folds 0+4 test. Earlier chain files are kept (`chain_mtcnn5fix.npz` 11 steps, `chain_cvw3.npz` 10 steps before the CPSAM member).
- OOF file: `experiments/lane_hex_20261108/preds/chain_seedaudit10.npz` (a copy of `chain_audit_g_none.npz`).
  - `final_hex`: N = 8954 rows (8952 scored; rows 4495 and 4496 have HEX label 0), in `best_pipeline_20261106` row order, with `idx` and `fold`.
  - `c0`..`c4`: the pre-stretch candidate of each outer fold.
- Reproduce the OOF number (CPU, about 3 min). `CHAIN_FORCE=1` keeps every listed step (the audit rule decided them):
  ```
  cd /workspace/clear-ec-research/experiments/lane_hex_20261108
  CHAIN_FORCE=1 CHAIN_NORESULTS=1 CHAIN_OUT=chain_repro /workspace/venv/bin/python source/chain.py ap1 h6w graphens imgstats prior ridgeev ridgeev_fall16 segm cvw3m_fix mtcnn5_fix
  ```
  The last step line must read `"step": "mtcnn5_fix", "f04": 9.8644, "all5": 9.7326` (9.86437 / 9.73258 unrounded).
- Deploy and nested settings: `CHAIN_FORCE=1 source/chain_deploy.py <same steps>` writes `preds/deploy_params.json` (the 11-step version is `archive/deploy_params_11step_chain_mtcnn5fix_20261110.json`).
  - It replays the same nested chain (same numbers).
  - It stores, for every step, the per-fold nested pick and fit (`nested`) and the per-fold stretch (`nested_stretch`).
  - It also stores the deploy pick and fit on all five folds (`deploy_pick`, `params`, `stretch`).
- Test-time entry point: `source/hex_apply.py`, function `apply(steps, stretch, inputs)`.
  - Deploy: `deploy_steps(P)` with `P['stretch']`.
  - Nested fold f: `nested_steps(P, f)` with `P['nested_stretch'][f]`.
- **Parity (verified 2026-11-11 06:40 EST for the 10-step chain):** `python source/hex_apply.py oof` builds the stored OOF inputs (`preds/hex_inputs_oof.npz`; cvw3 = mean log over seeds 46 and 47). Each fold's rows go through that fold's nested parameters.
  - It reproduces the OOF number exactly: HEX **9.7326** on five folds and **9.8644** on folds 0+4 (8952 labelled rows).
  - Max |difference| against `chain_seedaudit10.npz` `final_hex` is 8.9e-16.
  - Per fold: 10.0660, 9.5032, 9.8515, 9.5798, 9.6626. Output: `preds/hex_apply_oof.npz` (`hex_nested`).
  - The same inputs through the deploy parameters (fit on all five folds, so in-sample) give 9.7153 (`hex_deploy_in_sample`). That is not an OOF number.
  - The cell-route lane reproduced the chain independently on all 1791 fold-0 rows to 1.3e-15 (11-11, its notes 06:30 EST).

## Seed audit (coordinator, 2026-11-11)

Rule fixed before any read: a trained step resting on one seed gets a second seed on five folds (same recipe, only the seed changes) and is read with the mean over seeds. If the chain with the seed mean is no worse than the chain without the step (five folds), the step stays and deploy uses every seed with a full-data build by 15:00 EST; otherwise it drops. No seed is picked. All values are at the end of the chain, every other step kept.

| Step | Without the step | Seed 1 | Seed 2 | Seed mean | Decision |
|---|---:|---:|---:|---:|---|
| `cvw3` (CV-weighted DINOv3-B; s46 = the CV lane's arms, s47 = `runs/cvw3_f{f}_s47`) | 9.73308 / 9.85886 | s46 9.73579 (+0.00271) / 9.86515 (+0.00629) | s47 9.73020 (-0.00288) / 9.86121 (+0.00235) | 9.73222 (-0.00086) / 9.86223 (+0.00337) | stays (seed mean); folds 0+4 disagree |
| `gnn:hex` (5 training seeds; detector seed pre46 vs pre47 graphs), on the chain with the cvw3 mean | 9.73258 / 9.86437 | pre46 9.73222 (-0.00036) / 9.86223 | pre47 9.73258 (+0.00000; nested pick `none` on every fold) / 9.86437 | 9.73261 (+0.00003) / 9.86229 | drops (worse by 3e-5); folds 0+4 disagree |
| `mtcnn5` (CPSAM 512 multi-target) | a fold takes 8.6 h; the second seed runs on folds 0 and 4 only (GPU 3 and GPU 0, landing about 14:20 / 14:40 EST); a two-fold check that does not change the deploy model | | | | stays as is (one seed) |

Values are five folds / folds 0+4. Why cvw3 flipped: it was kept against the 9-step chain, and the CPSAM member added later overlaps with it (mtcnn5 gains -0.0074 / -0.0121 without cvw3, -0.0022 / -0.0053 with the single-seed cvw3). Every other step is multi-seed or has nothing trained: `ap1` (blend weights of the best-pipeline members), `h6w`, `prior`, `ridgeev` (3 pre-only detectors s46/47/48), `graphens` (10/3/3 seeds), `ridgeev_fall16` (16 detectors), `imgstats` and `segm` (nothing trained).

Leave-one-out on the 10-step chain (information only; coordinator: no cascade, the moves are about 0.001 and removing steps on them would be choosing on the reported rows):

| Removed | Five folds | Folds 0+4 |
|---|---:|---:|
| `segm` | +0.02520 | +0.01067 |
| `imgstats` | +0.01620 | +0.01693 |
| `ap1` | +0.00967 | +0.01132 |
| `prior` | +0.00639 | +0.00488 |
| `h6w` | +0.00507 | +0.00243 |
| `graphens` | +0.00435 | +0.00166 |
| `mtcnn5_fix` | +0.00230 | +0.00694 |
| `cvw3m_fix` | +0.00144 | -0.00185 |
| `ridgeev_fall16` | +0.00097 | -0.00121 |
| `ridgeev` | -0.00046 | -0.00227 |

Without `ridgeev` the chain reads 9.73212 / 9.86210, and then `graphens` turns negative on five folds (-0.00108; +0.00083 on folds 0+4). Logs: `logs/chain_audit_*.log`, `logs/loo_f10.summary`, `logs/loo_c9.summary`.

### Lighter alternatives (historical: read on the pre-CPSAM 10-step chain on 11-09; coordinator chose the full chain)

| Chain | Five folds | Folds 0+4 | What it needs beyond the light core |
|---|---:|---:|---|
| Full, 10 steps (then) | 9.7380 | 9.8704 | peaks of 16 detectors on every test image, CV-weighted DINOv3 |
| Without `ridgeev_fall16` (`preds/chain_light1.npz`) | 9.7383 | 9.8684 | CV-weighted DINOv3 |
| Without `ridgeev_fall16` and `cvw3_fix` (`preds/chain_light2.npz`) | 9.7421 | 9.8694 | nothing extra |

- `ridgeev_fall16` needs consensus peaks of 16 detectors on every test image: 3 pre-only, 9 `sg2` specialists and 4 frozen heads. All 16 have full-data versions (8 in `green_track_20261028/deploy/detectors/`, 8 built in this lane on 11-10).

## Formula (per image, HEX column, in step order)

Notation:
- `lc` is the running log HEX.
- `base = exp(0.7 log P1 + 0.3 log A)`.
- `h6 = -0.2084 + 0.5989 f6`, used only where `have` holds.
- The deploy pick of every step is listed; in brackets, the OOF picks on folds 0-4 where they differ.

| # | Step | Deploy pick | Operation |
|---|---|---|---|
| 1 | `ap1` | w 0.7 (fold 4: 0.8) | `lc = log(exp(0.7 log P1 + 0.3 log A))`. Where `have`: `lc = 0.72 lc + 0.28 h6`. Then `lc += BEST_hex`. |
| 2 | `h6w` | w 0.16 (folds 3, 4: 0.22) | where `have`: `lc += (0.16 - 0.28) (h6 - log base)` |
| 3 | `graphens` | s10x3 (fold 2: current) | `lc += G_s10x3 - GRAPH_hex` |
| 4 | `imgstats` | ridge, alpha 1000 (folds 0, 1, 4: 10) | features: log(mean+1), log(contrast+1), black_frac, log(focus+1) |
| 5 | `prior` | `cm1.0_h6c`, ridge alpha 10 | feature: log cm1.0_h6c |
| 6 | `ridgeev` (k3) | all 5 features, ridge alpha 1000 (fold 1: 10) | features: log h6_s15, h6_s25, h6_s35, h6_dark; darkness |
| 7 | `ridgeev_fall16` | `h6_s15`, ridge alpha 10 | feature: log h6_s15 of the 16-detector consensus graphs |
| 8 | `segm` | group `bh`, ridge alpha 100 | only frames with the first 15 segmentation features finite and n_int >= 50; other frames unchanged |
| 9 | `cvw3m_fix` | fixed share 0.2 | `lc += 0.2 (log D_cvw3 - log base)`, `D_cvw3` = exp of the mean log over seeds 46 and 47 (each the mean log over epochs 8-12) |
| 10 | `mtcnn5_fix` | fixed share 0.2 | `lc += 0.2 (log D_cps5 - log base)`: the CPSAM 512 member (recipe multi-target loss, no SSL), mean log over epochs 8-12 |
| 11 | stretch | fit on all rows | `HEX = 0.99012 exp(1.0 (lc + 0.61349) - 0.61349)` |

How the ridge steps work:
- Inputs are `[features..., lc]`.
- Missing features are filled with `feature_median`. `segm` has no fill: it acts on usable frames only, and `h6_strict` NaN is set to 0.5510 (`h6_strict_fill`).
- `z = (x - mean) / sd` and `lc += z . w + intercept`.
- All numbers are in `preds/deploy_params.json`, fit on the 8932 labelled rows that are not leaked.
- `segm` group `bh` has 16 features: log h6, h5, h7 (each +1e-3); log(n_int + 1); log area_cv, log cd_mm2; coverage; the n3..n9 side-count histogram as shares; log h6_strict.

## Test-time path (per image, label-free)

`hex_apply.apply(deploy_steps(P), P['stretch'], I)` with `P = preds/deploy_params.json`. The inputs `I` per image:

| Input | How to get it at test time |
|---|---|
| `A`, `P1` | recipe A HEX (s46 + s47, TTA at epoch 20, times the all-rows scalar 0.99159 in `lane_cv_20261108/deploy/cv_deploy_params.json`) and the Phase-1 HEX, as in `hybrid_oof` |
| `f6`, `have` | green hex6 feature (`green_track_20261028/deploy/hex6_correction.py`, 10 detectors) |
| `graph` | the CV lane's all-rows GRAPH, HEX column: mean over `gfull_{noctx,pre47,pre48}` of the mean over seeds (`lane_cv_20261108/source/cv_image_parts_20261110.py`, `GraphNets(...).out(graphs)`) |
| `best` | `graph + CA.ridge(P_cv, [emb_s0, emb_s1, emb_s2], 'HEX')`: the simulation-embedding ridge, `lane_cv_20261108/deploy/cv_ridge_params.npz` (HEX strength 3000) |
| `g_s10x3` | mean of three HEX columns: the CV lane's `gfull_noctx` (10 seeds) and this lane's `deploy/graph_models/full_d64_l5_pre47_s10` and `full_d64_l5_pre48_s10`. Run them with `gnn_hex_infer.predict(name, g_pre47/48.npz)` |
| `img`, `prior_h6c`, `ridge_k3`, `ridge_fall16_h6_s15`, `seg` | `hex_test_features.build_arrays(imgs, peaks, cd, outd, do_seg=True)` |
| `dino_cvw3` | `exp(family_infer.predict([deploy/cvw3_f99_s46/... and deploy/cvw3_f99_s47/runs/fold99/dinov3_b16/epoch_{08..12}.pt], ...)[:, 2])`: the mean log over all 10 checkpoints, equal to the mean of the two seeds' means |
| `cps5` | `exp(family_infer.predict(deploy/mtcps_c512_f99/runs/fold99/cpsam_c512/epoch_{08..12}.pt, ...)[:, 2])` (verified exact on the fold-4 run, 16 frames x 5 checkpoints) |

`g_hex46` (the HEX-only graph net) is no longer an input: `gnn:hex_pre46_s5` was dropped by the seed audit.

Details of the `build_arrays` row:
- `cd` = hybrid-B CD = 0.5 A_cd + 0.5 P1_cd. It equals `hybrid_oof` B exactly, arithmetic not geometric.
- `peaks` are single-view peaks from `det_peaks_image.py`: SG.heat, then SG.peaks2 with threshold 0.2 and NMS 12 px.
  - pre3 = `green_track_20261028/deploy/detectors/sg2pre_s{46,47,48}.pt`, the pair the CV and cell-route lanes use.
  - all16 adds the green track's `sg2_s46..50` and the 8 full-data detectors built in this lane's `deploy/`.
- The graphs `g_pre47/48/fall16` come out of the same call.

Verified on training frames:
- The features equal the stored training features on 40 pseudo-test frames; segmentation is within 1.1e-4.
- The detector peaks, `family_infer`, and `hex_apply` with nested parameters are exact.

## Inputs and trained components

| Input | Used by | OOF file | How it is made | Full-data training (time) |
|---|---|---|---|---|
| A, P1, f6, `have`, BEST, GRAPH | 1-3, 9, 10 | `recipeA_analysis_20261028/hybrid_oof.npz`; `submission2_search_20261103/green_features.npz`; `meta_ceiling_20261103/sweep/sim_pred/sim2_l8_ongraph_nested.npz`; `sweep/graph_sweep/d64_l5_{noctx_s10,pre47_s3,pre48_s3}.npz` | the best pipeline's shared parts (`meta_ceiling_20261103/source/best_pipeline_20261106.py`) | as for the other lanes (CV lane candidate); not changed here |
| G_s10x3 | 3 | `sweep/graph_sweep/d64_l5_{noctx_s10,pre47_s10,pre48_s10}.npz` (key `f{f}`, column HEX) | `meta_ceiling_20261103/source/cell_graph_sweep3_20261105.py`, d 64, layers 5, epochs 25, lr 1e-3, wd 0.01, drop 0.1, 10 seeds, no ctx, graphs `sweep/cell_graphs{,_pre47,_pre48}.npz` | built (see Deploy builds); noctx is the CV lane's `gfull_noctx` |
| image statistics | 4 | `eval_harness_20261031/image_stats.npz` | whole-frame mean, contrast, black fraction, focus | none |
| prior `cm1.0_h6c` | 5 | `lane_hex_20261108/feats/prior_pre3.npz` | `source/feat_prior.py feats/prior_pre3.npz pre3` (peaks of the 3 pre-only detectors and the hybrid-B CD) | none |
| ridge evidence k3 | 6 | `lane_hex_20261108/feats/ridge_k3.npz` | `source/graph_prep_lane.py g_k3.npz ...`, then `source/feat_ridge.py feats/g_k3.npz k3` | none |
| ridge evidence fall16 | 7 | `lane_hex_20261108/feats/ridge_fall16.npz` | `source/fuse_peaks.py feats/fpk_all16 all16`, then `source/graph_prep_lane.py g_fall16.npz feats/fpk_all16 3 0.25`, then `source/feat_ridge.py feats/g_fall16.npz fall16` | 16 full-data detectors (built); 16 detector passes per test image |
| segmentation | 8 | `lane_hex_20261108/preds/seg_all.npz` | zero-shot Cellpose-SAM (`cpsam_models/cpsam_v2`, no training): `CP_FLOW=0.8 CP_PROB=-2 CP_CLAHE=1 source/seg_run.py seg_loose`, then `source/seg_merge.py seg_loose seg_all` | none |
| D_cvw3 | 9 | s46: `lane_cv_20261108/arms/dinov3_cvw3_f{0..4}_s46/...`; s47: `lane_hex_20261108/runs/cvw3_f{0..4}_s47/runs/fold{f}/dinov3_b16/predictions_epoch_08..12.npz` | `CD_LOSS=cvw3 source/family_cvw3_copy.py all --family dinov3_b16 --fold f --seed {46,47} --epochs 12 --schedule-epochs 10 --micro 16 --ssl-from <the fold's s46 SSL>` (s47 queue: `source/seed_audit_queue.sh`) | s46 2500 s, s47 2456 s |
| D_cps5 | 10 | `runs/mtcps_nossl_c512_f{0..4}/runs/fold{f}/cpsam_c512/predictions_epoch_08..12.npz` | `source/mtcps.py all --family cpsam_c512 --fold f --ssl-epochs 0 --epochs 12 --schedule-epochs 10` | 36079 s |

## Deploy builds (11-10 and 11-11, GPU 1)

| Component | Output | Measured time |
|---|---|---|
| 4 sg2 detectors (s51, s52, sigma 3.5, sigma 5) | `deploy/detectors/<name>/detectors/sg2_s*.pt` | done: 385-396 s each |
| 4 frozen heads on the full-data encoder (sg2f s46-48, sg2pref s46) | `deploy/frozen_heads/<kind>_s<seed>/fold0/head.pt` | done: sg2f 195-205 s each, sg2pref 670 s |
| graph nets pre47_s10, pre48_s10 (10 seeds each) | `deploy/graph_models/full_*_s{seed}_e{21..25}.pt` | done: 803 s, 811 s. `gnn_hex_infer --check` max diff 1.3e-8, 1.2e-8. (The HEX-only pre46 build, 401 s, is no longer used.) |
| CV-weighted DINOv3 (cvw3) seed 46, fold 99 | `deploy/cvw3_f99_s46/runs/fold99/dinov3_b16/epoch_{08..12}.pt` | done: 2500 s. Sanity on 64 training frames: HEX r 0.978 with the OOF member |
| CV-weighted DINOv3 (cvw3) seed 47, fold 99 | `deploy/cvw3_f99_s47/runs/fold99/dinov3_b16/epoch_{08..12}.pt` | done 11-11 06:53 EST: 2456 s (with other jobs on the pod). Sanity on 64 training frames: s47 HEX r 0.978 with its OOF member (mean log ratio -0.0076); the 10-checkpoint deploy input (s46 + s47) r 0.980 with the OOF seed mean (mean log ratio -0.0039) |
| CPSAM 512 multi-target no-SSL, fold 99 | `deploy/mtcps_c512_f99/runs/fold99/cpsam_c512/epoch_{08..12}.pt` | done: 36079 s (10.0 h). Sanity on 64 training frames: HEX r 0.954 with the OOF member |

Rebuild cost of the whole HEX-specific set on one GPU: detectors about 45 min, graph nets about 27 min, cvw3 42 min (s46) + 41 min (s47), CPSAM 10.0 h. Each component fits the 12 h limit.

## Not in the candidate (tested on top, no gain)

All are listed in `RESULTS.tsv`:
- `gnn:hex_pre46_s5` (dropped by the seed audit, see above);
- segmentation variants: median-filled, h6 blend, boosted trees, centroid-Delaunay (`segv`), the rotated second run (`segm2`, `segr`), the graph network on segmentation masks, and the Vigueras-Guillen TVST edge network (`segvig`, closed 11-11);
- the HEX-weighted CPSAM loss;
- graph-network variants on detector graphs, ridge evidence from other detector sets, and family members on folds 0 and 4 (including the backbone lane's C-Mixup and attention-pool DINOv3-B, 11-11);
- the shipped DINOv3 at share 0.2.

## Downstream use

- The cell-route lane fits one more HEX step on top of this chain's `final_hex`. Refit on `chain_seedaudit10.npz` (11-11): **HEX 9.72623** on five folds vs the chain 9.73258 (-0.00635), **9.8554** on folds 0+4 vs 9.86437; per fold -0.017 / -0.008 / -0.008 / +0.001 / -0.001 (`lane_cellroute_20261109/out/corr_hex_on_seedaudit10.npz`). On the 11-step chain it read 9.7283 vs 9.7358. Its 20-frame no-stub deploy run with this chain: HEX median / max |rel| 1.58 / 7.0 % against OOF (training frames, not a parity number).
  - Files: `lane_cellroute_20261109/out/deploy_hexkonan_params.json`, `source/deploy_hexkonan.py`.
  - It reuses this lane's loose Cellpose-SAM segmentation (`preds/seg_all.npz`, `seg_run.py`).

## CPSAM member (kept 11-10)

- Five-fold read at share 0.2 on the 10-step chain of 11-10: -0.0053 on folds 0+4, -0.0022 on five folds. In the audited chain, removing it costs +0.0023 / +0.0069.
- OOF members: `runs/mtcps_nossl_c512_f{0..4}/runs/fold{f}/cpsam_c512/predictions_epoch_08..12.npz`.
- Recipe: `source/mtcps.py`, the lane GPU 3 family recipe, recipe loss, no SSL, epochs 12, schedule 10. Fold 2 ran with the CD lane's fix for its two HEX-zero validation rows, which changes scoring only.

## Reproduction check (10-step chain, 2026-11-11 06:40 EST)

- `CHAIN_FORCE=1 chain_deploy.py` with the 10 steps above (`logs/chain_deploy_seedaudit10.log`) printed the same per-step values as `chain_audit_g_none` and the deploy picks in the table.
- `hex_apply.py oof` printed nested 9.7326 / 9.8644, max |difference| 8.9e-16 against the chain; in-sample with the deploy parameters, 9.7153.
