# CV lane final summary (2026-11-13)

Lane folder: `/workspace/clear-ec-research/experiments/lane_cv_20261108` (pods `clear-ec-cpu-1109`, `clear-ec-research-4gpu-1104`).
CV MAPE throughout. "Strict" means the cv_search procedure: for each outer fold, selection, weights and the inner stretch use the other four folds only.
Baseline: best pipeline (`meta_ceiling_20261103/source/best_pipeline_20261106.py`, REF_BEST) CV 10.1512, five folds, as computed.
State at close: no GPU job, waiter, loop or scheduled wake-up of this lane is running (checked 21:43 EST 11-13).

## 1. Outcome

- **[D-CV-001] CV freeze.** Six-component log-correction stack on the best pipeline:
  - plain DINOv3-B (seeds 46/47/49, w 0.125);
  - frozen token-statistic heads: spectrum 0.25, cov_sqrt 0.375, autocorr 0.25;
  - plain TTA-peak graph quartet 1.0;
  - order-loss graph quartet 0.25.
- **Score.** Five folds: 10.1101 as computed, 10.1160 strict, against the best pipeline's 10.1512 (−0.0352 strict). Verifier reproduced 10.11596 [O-VER-001].
- **[O-CV-L5-001] L5.** The nested code with no outer fold reproduces `deploy/cv_deploy_params.json` (md5 3b437500) exactly.
- **[D-PKG-002/003] Tiers.** All strict five folds, with the incumbent 10.1160 recomputed in the same code [O-CV-TIERS-001, O-VERIFY-CV-TIERS]:
  - **deploy_t1** (dino = s46b, e16-20, single view): 10.1115.
  - **deploy_t2**: byte-identical to t1.
  - **deploy_t3** (dino = s46b epoch 20 only): 10.1138.
  - **deploy_t3_alt** (no DINOv3 term): 10.1238.
  - L5 passes for each.
- **Packages.** Tier p uses deploy_t1 [O-DEPLOY-017]; tier t3 uses deploy_t3 [A-DEPLOY-013].
- **Integrated strict mean** [O-VER-003, D-PKG-003]: full 8.6679; p 8.6682. The best pipeline with scalars refit reads 8.7363.
- **[F-TESTGAP-001] Submission 1 test gap.** The test-minus-OOF pattern is inside 1000-image sampling noise (joint tail 0.40). No HEX adjustment is safe.

## 2. Models

Paths are relative to `/workspace/clear-ec-research/experiments/`. Sizes come from my records; nothing was measured on DDN for this file.

| path | role | what | size |
|---|---|---|---|
| lane_cv_20261108/graph_models/gfull_{noctx_s0..9,pre47_s0..2,pre48_s0..2}_s<k>_e{21..25}.pt | deployed: full, **p**, **t3** | all-rows GRAPH base nets (16 seeds × 5 snapshots) | 80 files, 40.5 MB |
| lane_cv_20261108/graph_models/full_tta_{s46,s47,s48,seeds3}_tta4_s{0,1,2}_e{21..25}.pt | deployed: full, **p**, **t3** | plain TTA-peak graph quartet, full data | 60 files, 30.5 MB |
| lane_cv_20261108/graph_models/gord_{s46,s47,s48,seeds3}_tta4_full_s<k>_s<k>_e{21..25}.pt | deployed: full, **p**, **t3** | order-loss graph quartet, full data | 60 files, 30.4 MB |
| lane_cv_20261108/graph_models/*_fullfit_pred.npz | deploy-fit input | in-sample predictions of the all-rows nets (read by cv_deploy_fit) | small |
| lane_cv_20261108/frozen_heads_full/{std,spectrum,cov_sqrt,autocorr}_s{0,1,2}_e{16..20}.pt | deployed: full, **p**, **t3** | frozen token-statistic heads on the recipe A s46 e20 encoder | 60 files, 100.5 MB |
| lane_cv_20261108/arms/dinov3_b16_f99_s47, _f99_s49 (epoch_16..20.pt) | deployed: full only | full-data plain DINOv3-B seeds 47/49 for the 3-seed CV term | 10 files, 3.43 GB |
| lane_cd_20261108/arms/dinov3_b16_f99_s46b (CD lane's) | deployed: full, **p** (e16-20), **t3** (epoch_20 only) | the CV term's seed 46 | 5 files, 1.72 GB (e20 alone 0.343 GB) |
| meta_ceiling_20261103/sweep/sim_pred/sim2_l8_s{0,1,2}.pt | deployed: full, **p**, **t3** | simulation-embedding nets for the CV/HEX ridge | 3 files, 2.3 MB |
| lane_cv_20261108/arms/dinov3_order_f99_s{46,47,49} | closed but reusable | full-data order-loss DINOv3-B (dropped by the 3-seed rule) | ~1.72 GB each |
| lane_cv_20261108/arms/dinov3_b16_f{0..4}_s49 | candidate OOF source | fold models behind D-CV-001's seed-49 term | 5 runs |
| lane_cv_20261108/arms/dinov3_cvw3_f{0..4}_s46, f0_s47 | closed for CV; **read by the HEX lane** (cvw3 column) | CV-weighted-loss DINOv3-B fold runs | – |
| lane_cv_20261108/arms/dinov3_order_f{0..4}_s{46,47,49}, dinov3_b16_f*_s50, dinov3_aug_*, dinov3_sd_*, dinov3_lds_*, dinov3_lw_*, dinov2_order_*, vitl16_order_* | closed | fold runs of closed ideas (section 4b) | – |
| lane_cv_20261108/deploy, deploy_t1, deploy_t2, deploy_t3 | parameter sets (not weights): full / **p** + t1 / t2 / **t3** | cv_deploy_params.json, ridge params, in-sample members, OOF and inputs for parity | small |

## 3. Data

| path (lane_cv_20261108/ unless stated) | what | regenerable from | other readers |
|---|---|---|---|
| out/pool_20261111.npz | 25 five-fold CV component vectors (CV column, per fold model) | cv_stack5_20261108.py with STACK_DUMP | verifier (CD lane) |
| out/cv_final_20261110.npz + _comps.npz | D-CV-001 as-computed OOF (10.1101), per-fold weights and w_all | cv_stack5 with --save and SAVE_COMPS | cv_deploy_fit; deploy lane tests |
| out/search_fixed6_20261111.npz | D-CV-001 strict OOF (10.1160) | cv_search_20261111.py | verifier |
| out/cv_tier_{incumbent,t1,two_seed,t3_e20,t3_nodino,t1_noquart,t3_fzonly}_20261112.{npz,json} + _comps | tier strict OOFs, weights, deploy weights | cv_tiers_20261112.py (7 s each) | verifier (verify_cv_tiers_20261112.py) |
| out/l5_cv_tier_*.json, out/l5_deploy_check_20261111.json | L5 evidence | l5_tier_check_20261112.py, l5_deploy_check_20261111.py | verifier |
| deploy*/cv_inputs_oof.npz, cv_final_oof.npz | per-row OOF inputs plus per-fold parameters (cv_apply parity) | cv_deploy_fit_20261110.py inputs / ids | deploy lane (test_cv_fold0, recipe_infer checks) |
| reads/own/*.npz | frozen-head OOF log predictions, folds 1-3 (folds 0/4 in lane_gpu2_20261106/reads/own) | frozen_read (lane_gpu2 source) on frozen_own features | cv_stack5 fz_comp |
| frozen_own_f{1,2,3}, frozen_own_f99 | token-block features (recipe A encoders) | token_stats_cv_20261108.py own <f> (~35 min GPU each) | frozen_full_heads (refit) |
| graph_runs/*.npz | graph member OOF per fold (adopted: tta_s4{6,7,8}_tta4_5f, tta_seeds3_tta4_5f, gord_*_5f; the rest are variants) | graph_train_cv / graph_order_cv (GPU or CPU hours) | cv_stack5 graph_comp |
| graphs/*.npz | cell-graph caches from detector peaks (TTA and variants) | graph_prep_sp_20261108.py | graph training only |
| out/tta_peaks/{s46,s47,s48}_tta4 | flip-TTA detector peak caches | tta_infer_20261108.py | deploy lane scripts (2 refs) |
| out/member_tta/ | TTA predictions of cvw3 / DINOv3 s49 members (closed for CV) | tta_infer | possibly the HEX lane (cvw3); check before moving |
| reads/hex_test_gap_20261111.{md,json} | test-gap analysis | hex_test_gap_20261111.py | coordinator, PI |
| external inputs (not this lane's) | recipeA_analysis_20261028/hybrid_oof.{npz,json} (A and LOFO scalars), eval_harness_20261031 (labels, folds, donors, image_stats), submission2_search_20261103/green_features.npz (BP good mask, f6), meta_ceiling_20261103/sweep/{graph_sweep,sim_pred} (GRAPH, BEST) | owners' scripts | everyone |

## 4. Ideas

(a) Adopted (in D-CV-001)
- **idea-cv-frozen-token-statistic-heads**: spectrum / cov_sqrt / autocorr heads on the recipe A encoder. Five-fold stack 10.139 alone (#15). Dropping them costs +0.0224 (as computed, price list).
- **idea-cv-tta-peak-graph-quartet**: graphs on flip-TTA detector peaks, threshold 0.35, four members. TTA-graph trio alone 10.145 vs 10.151. Dropping both quartets costs +0.0118 strict (t1 10.1115 → 10.1233).
- **idea-cv-noise-weighted-order-loss (graph quartet)**: added as a second graph component at w 0.25. Final stack 10.1101 vs 10.1143 without it (as computed). The order quartet in place of the plain one read −0.0265.
- **idea-cv-dinov3-b16-member-seed-mean**: plain DINOv3-B term, seeds 46/47/49 in the freeze; deploy t1 uses one seed (10.1115 vs 10.1160 strict). Dropping the term costs +0.0078 strict.
- **process-deploy-equals-nested-no-outer-fold (L5)** and **process-strict-nested-stack**: the inner-stretch leak was worth 0.006 (10.1101 vs 10.1160).

(b) Closed, with the number that closed it (vs the stack of the time, five folds unless stated)
- **idea-cv-order-loss-dinov3-member**: pre-registered 3-seed read 10.1143 vs 10.1101 [D-CV-001].
- **idea-cv-vitl16-member**: weight 0 on five folds. **idea-cv-vith16-member**: weight 0 on folds 0+4.
- **idea-cv-weighted-loss-dinov3**: −0.010 alone, weight 0 next to the recipe member. **idea-cv-weighted-loss-frozen-heads**: weight 0.
- **idea-cv-dinov3-augmentation-bundle**: weight 0; inside the seed mean −0.030 vs −0.037. **idea-cv-dinov3-four-seeds**: −0.031 vs −0.037. **idea-cv-dinov3-flip-tta-member**: weight 0. **idea-cv-legible-window-crop-dinov3**: worse (+0.031 at w .25, folds 0+4).
- **idea-cv-frozen-heads-on-dinov3-tokens**: stack −0.024, worse than the member alone (−0.031).
- **idea-cv-graph-variants** (vs the shipped graph trio): spacing-prior NMS +0.010; window attention +0.013 / +0.033; 5 windows +0.009; fine-tuned-detector graphs +0.031 / +0.037; whole-frame graph +0.005; 80 epochs +0.116; 5-detector ensemble weight ~0.
- **idea-cv-segmentation-features**: CPSAM features +0.011; Vigueras-Guillen features +0.0196; Konan weighted-bisector CV from peaks +0.015 to +0.017.
- **idea-cv-residual-from-hex-cd-columns**: +0.135.
- **idea-cv-sd-target-member**: −0.0358 vs −0.0411 (worse). **idea-imbalanced-regression-lds-member**: weight 0 (folds 0+4).
- **idea-cv-order-loss-vitl16 / -dinov2 / -graph-sets**: weight 0 / weight 0 / stack −0.0389 vs −0.0460.
- **idea-cmixup-dinov3 / idea-attention-pool-dinov3**: worse (−0.0622 vs −0.0695) / weight 0 (folds 0+4).
- **idea-cv-from-scratch-combination-search**: grid 10.1176 and forward selection 10.1324 vs fixed six 10.1160 (strict); CPSAM multi-target and window members got weight 0 there.
- **idea-synthetic-pretraining (CV)**:
  - zero-shot corrector +0.0152 [O-BB-SYN-ZS];
  - synthft3 DINOv3-B in place of plain +0.0035 [O-CV-SYN-002];
  - recipe A swap +0.0041 [O-CV-SYN3-001];
  - synthetic ViT-L +0.0078 [O-CV-SYN4-001];
  - all strict five folds. The folds 0+4 screens (−0.011, −0.0154) did not hold (L2).
- **idea-hex-test-gap-adjustment**: an offset fix has negative expected value with unknown sign (+0.35 to +0.42) [F-TESTGAP-001].

(c) Open or only partly tested
- **idea-cv-quartet-free-stack-with-shared-flips**: whether HEX needs the flipped pre-only peaks was not checked. If it does not, dropping the quartets saves at most 3 of 4 detector flips for +0.0118.
- **idea-test-hex-label-mix-shift**: the only covariate shift that fits the test pattern plausibly (KL 0.006; HEX ≥ 0.66 ×1.27). Unobservable without test labels.
- **idea-cv-fp16-path-impact**: the T4 fp16 path deviates CV by median 0.175 % / max ~0.9 % [O-DEPLOY-020/021]. No CV-specific score impact was measured by this lane.

## 5. Folder map

Lane folder `lane_cv_20261108/`:
- source/ — all lane code. **KEEP.** By-name imports from other lanes:
  - lane_hex_20261108/source: chain.py, joint_fit.py, joint_deploy.py, hex_apply.py, graph_train_full.py, cvw3 scripts;
  - lane_cellroute_20261109/inference: recipe_infer.py, params_build.py, test_*.py, impact*.py;
  - lane_cd_20261108 verify_cv_*.py.
  - Modules imported this way: graph_prep_sp_20261108, graph_train_cv_20261108, tta_infer_20261108, token_stats_cv_20261108, cv_apply_20261110, cv_image_parts_20261110.
- graph_models/ — **KEEP-MODEL** (p, t3, full). Its _archive_wrong_seed_20261110/ is **DELETE-NOW** (superseded wrong-seed weights).
- frozen_heads_full/ — **KEEP-MODEL** (p, t3, full).
- deploy/ — **KEEP** (full tier; manifests/deploy.json and params/deploy.json name it). deploy/_superseded/ is **ARCHIVE**.
- deploy_t1/, deploy_t3/ — **KEEP**, named by manifests/params deploy_p, deploy_t1, deploy_t3.
- deploy_t2/ — **KEEP**: named by deploy_t2.json; byte-identical to t1.
- deploy_t3_alt/ — **ARCHIVE**: no package uses it.
- deploy_fallback_10.1101/ — **DELETE-AFTER-SUBMISSION**: duplicate of deploy/ plus copies of the frozen_* folders.
- arms/dinov3_b16_f99_s47, _s49 — **KEEP-MODEL** (full tier only; manifests/deploy.json).
- arms/dinov3_b16_f{0..4}_s49 — **KEEP** (reproduce D-CV-001).
- arms/dinov3_cvw3_* — **KEEP** (the HEX lane reads `arms/dinov3_cvw3_f{f}_s46`).
- arms/dinov3_order_f99_s46 — **KEEP until the manifests are cleaned**: manifests deploy, deploy_p and deploy_t3 still list it under orderdino_cv (5 entries each), although no tier's components use it.
- arms/dinov3_order_f99_s47/s49 and arms/dinov3_order_f* — **DELETE-AFTER-SUBMISSION** (closed). `arms/dinov3_order_f0_s46` is named by deploy-lane test scripts.
- arms/dinov3_b16_f*_s50, dinov3_aug_*, dinov3_sd_*, dinov3_lds_*, dinov3_lw_*, dinov2_order_*, vitl16_order_* — **DELETE-AFTER-SUBMISSION** (closed fold runs).
- arms/_failed_nan_dinov3_b16_f99_s47, _s49 — **DELETE-NOW** (partial runs that died on the NaN bug).
- arms/_archive/ — **DELETE-NOW** (smoke tests, stopped partials). arms/cvw — unknown leftover; check before deleting.
- graph_runs/ — **KEEP** (adopted quartet OOF plus small variant evidence). cv_stack5 names it as LO.
- graphs/ — **DELETE-AFTER-SUBMISSION** (graph caches, regenerable).
- out/ — **KEEP** (pool, stack, strict and tier OOFs, L5 json); out/tta_peaks — **KEEP** (deploy-lane scripts name it); out/member_tta — **ARCHIVE** (check the HEX lane first).
- reads/ — **KEEP** (reads/own feeds fz_comp; test-gap report).
- frozen_own_f{1,2,3}, frozen_own_f99 — **DELETE-AFTER-SUBMISSION** (regenerable features; f99 is needed only to refit the heads).
- frozen_dino_f{0,1,4}, frozen_dino_pca.npz — **DELETE-AFTER-SUBMISSION** (closed idea).
- logs/ — **ARCHIVE**. locks/ — **DELETE-NOW** (stale queue locks qC2-qC4).
- REPORT.md, RESULTS.tsv, FREEZE_CANDIDATE.md — **KEEP**.

Other folders this lane reads (`experiments/`):
- meta_ceiling_20261103 — **KEEP**: best_pipeline, stretch_nested and ceiling modules; sweep/graph_sweep (GRAPH); sweep/sim_pred (BEST, sim nets deployed). Every CV script hard-codes its source path in sys.path.
- eval_harness_20261031 — **KEEP**: labels, folds, donors, image_stats; imported by BP and hex_test_gap.
- recipeA_analysis_20261028 — **KEEP**: hybrid_oof.{npz,json} (A, LOFO scalars); named in cv_deploy_fit, cv_swapA_read and hex_test_gap.
- fulldata_recipeA_20261028 — **KEEP-MODEL**: recipe A f99 s46/s47 e20, the CV A term and the token encoder; cv_parts.token_encoder in all manifests.
- green_track_20261028/deploy/detectors — **KEEP-MODEL**: sg2pre_s46/47/48 (CV GRAPH, quartets, ridge).
- submission2_search_20261103/green_features.npz — **KEEP**: read by BP at import (good mask, f6).
- lane_gpu3_20261106/runs/dinov3_b16_f{0,4}_s46 and lane_gpu2_20261106/runs/dinov3_b16_f{0,4}_s47 — **KEEP**: DINOv3 term OOF; root() in cv_stack5 and cv_tiers.
- lane_gpu2_20261106/{frozen_own_f0,f4, reads/own, frozen_s46/pca.npz, source/frozen_read_20261106.py} — **KEEP**: frozen-head OOF and PCA basis.
- lane_cd_20261108/arms/dinov3_b16_f{1,2,3}_s{46,47} — **KEEP** (OOF). lane_cd_20261108/arms/dinov3_b16_f99_s46b — **KEEP-MODEL** (all tiers).
- phase2_foundation_20261002/runtime — **KEEP**: the PYTHONPATH the packages need [F-DEPLOY-002].
- submission1_test_20261030 — **ARCHIVE** (submission-1 smoke tests; the n60 rows were used in the test-gap deploy check).
- lane_backbone_20261108/runs/synthft* and lane_hex_20261108/preds/seg_all.npz — read-only here; owners decide.
- /workspace/{artifacts,checkpoints,logs,scratch_*} — not used by this lane; purpose unknown to me.

## 6. Loose ends

- **Stale metadata:**
  - Every CV params file (deploy/ and the tiers) carries a descriptive 'order_dino' path text from the fit template.
  - Manifests deploy, deploy_p and deploy_t3 list orderdino_cv (lane_cv arms/dinov3_order_f99_s46).
  - Neither is used: 'components' decides, and the p and t3 censuses contain no lane_cv arms files.
  - Moving that folder may still trip a manifest existence check.
- **t3 parameter folder in the Colab test:** the Colab t3 run used a copy of the t1 CV parameter folder [O-DEPLOY-023]. I have not checked which CV folder the box container test of t3 read. The t3 manifest names deploy_t3.
- **t1 and t3 are not gains.** The t1 and t3 strict differences (−0.0045, −0.0022) come almost entirely from fold 1: −0.036 / −0.034 there, +0.000 to +0.014 on the other folds. That is weight-selection noise; read them as "no loss" (L7).
- **Rule breach (self-report).** On 11-12 I ran `du --apparent-size` on two small lane folders (graph_models, frozen_heads_full) against the no-du-on-DDN rule. Nothing observable resulted.
- **Ledger corrections already posted:** O-CV-SYN-002b (link type); A-CV-SYN4-READ supersedes A-CV-SYN3-READ; O-CV-SYN4-001 supersedes O-CV-SYN2-001. No other correction is pending.
- **Waiter exit 143 (11-11 22:05 EST)** was my intentional stop of the old swapped-A waiter, by verified pid.
