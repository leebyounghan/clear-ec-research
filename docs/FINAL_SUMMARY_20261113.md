# CLEAR-EC Phase II: consolidated final summary (2026-11-13, 22:15 EST)

Merged by the coordinator from the five lane summaries (CD, CV, HEX, backbone, deploy / cell-route). The lane files hold the detail and stay the
source for every number here: `side-session-notes/local_5a5ca17b-018d-4ccd-b356-f8794b094404/final_summary_{cd,cv,hex,backbone,deploy}_20261113.md`
(copies should be placed in `docs/lane_summaries_20261113/` when the sessions are deleted). Ledger ids are in brackets; the ledger is
`ledger_20261111.jsonl` in the same notes folder.

Conventions: MAPE in %, five folds unless stated. "Strict" = everything fitted (weights, scalars, stretch, choices) is fitted without the scored
fold. Paths without a prefix are under `/workspace/clear-ec-research/experiments/` on pod `clear-ec-research-4gpu-1104`.

## 0. Where the project stands

| | mean | CD | CV | HEX | source |
|---|---:|---:|---:|---:|---|
| Submission 1, test (1,000 images) | 8.9675 | 6.250 | 10.143 | 10.510 | leaderboard |
| Submission 1, OOF | 8.877 | 6.312 | 10.204 | 10.115 | BRIEF |
| Best pipeline before the lanes, scalars refit per fold | 8.7363 | 6.2610 | 10.1479 | 9.7999 | BRIEF, [O-VER-002] (the lanes quote CV 10.1512 as computed and HEX 9.8026 strict as their own baselines) |
| Frozen candidate, full, strict | 8.6679 | 6.1743 | 10.1160 | 9.7134 | [O-VER-004] |
| **Tier p (built, tested, first to upload), strict** | **8.6682** | 6.1737 | 10.1115 | 9.7194 | [D-PKG-003] |
| Tier t3 (built, tested, fallback), strict | 8.6861 | 6.2138 | 10.1138 | 9.7308 | [O-DEPLOY-018] |
| Leaderboard 11-13: 1st / 2nd / new 3rd, test | 8.6920 / 8.7035 / 8.7560 | 6.049 / 5.975 / 6.089 | 9.928 / 10.060 / 9.988 | 10.099 / 10.075 / 10.191 | PI |

- Submission 2 is unused. Upload files for p and t3 are built and tested (section 1.3). Upload, Try-out and the submission are the PI's actions.
- Open risks before submitting: (a) tier p ran 287-288 s on one Colab T4 and 311-314 s on a second (limit 300 s; platform speed unmeasured)
  [O-DEPLOY-T4-COLAB, O-COORD-COLAB-T4-B]; t3 ran 173-174 s. (b) The p model archive is 15.06 GB against a possible 10 GB upload cap; t3's is
  8.81 GB [F-DEPLOY-006]. The Try-out note has the rule: p above 270 s exec time, or a refused model upload, means switch image and model to t3.
- All four GPUs were taken down on the PI's order at 21:41 EST 11-13 [D-GPU-002]. The order was lifted at about 22:30 EST for training-side
  CD attempts: a DINOv3-B fold-0 hyperparameter screen and sweep and two pretraining arms run on the CD lane's session
  [D-CD-TRAIN-SCREEN, D-CD-PRETRAIN-SCREEN, D-CD-SWEEP]; results are not in this file. The fellow-eye residual transfer (CD -0.236 with the ID
  as an oracle, +0.002 from the image alone) was rejected by the PI as a trick and is closed [O-CD-FELLOW-EYE]; the ViT-L averaged tracks and
  the recipe A epoch 9-12 window add nothing in the stack [O-CD-PART1-TRACKS-WINDOW].

## 1. Models

### 1.1 What each metric's stack is

| Metric | Decision | Stack | Strict | vs best pipeline |
|---|---|---|---:|---:|
| CD | D-CD-001 (tier t1 form in p) | log blend: Phase-1 five crop CPSAM members x SCALE, recipe A s46+s47, DINOv3-B/16 s46+s47 TTA, ViT-L/16 s46+s47, with shipped tier-p weights 0.25 / 0.40 / 0.10 / 0.25 (D-CD-001 as frozen for the full tier: 0.25 / 0.45 / 0.10 / 0.20); then mask density `cd_mm2` (n_int >= 50, w 0.1); then MAPE-optimal log-affine stretch (c 0.997542, g 1.075) | 6.1737 | -0.087 |
| CV | D-CV-001 (deploy_t1 in p) | best pipeline + six log corrections: plain DINOv3-B 0.125; frozen token-statistic heads (spectrum 0.25, cov_sqrt 0.375, autocorr 0.25); TTA-peak graph quartet 1.0; order-loss graph quartet 0.25 | 10.1115 | -0.040 |
| HEX | D-HEX-004 (t2 form in p) + D-HEX-005 | one nested ridge over all HEX corrections (38 columns in p; 39 with the CPSAM multi-target column in full/t1), incl. the attention-pool DINOv3-B column; then the cell-route step | 9.7194 | -0.083 |

### 1.2 Weights to keep (KEEP-MODEL)

The definitive lists are `lane_cellroute_20261109/package_20261112/<tier>/context/model_files.txt` (p: 443 files, 16.25 GB; t3: 384 files,
9.51 GB; full: 490 files, 29.98 GB).

| Folder | What | p | t3 |
|---|---|---|---|
| `deploy_review_20260905/staged/model` | Phase-1 five crop CPSAM members (5 x 1.22 GB) | yes | yes |
| `/workspace/clear-ec-research/cpsam_models/cpsam_v2` | Cellpose-SAM (backbone, zero-shot segmentation), 1.23 GB | yes | yes |
| `fulldata_recipeA_20261028/regression_std_s46, _s47` | recipe A full data; epoch_20 in p / t3, epochs 8-19 in full only | yes | yes |
| `lane_cd_20261108/arms/dinov3_b16_f99_s46b, _s47` | DINOv3-B/16 full data, epochs 16-20 | yes | s46b e20 only |
| `lane_cd_20261108/arms/vitl16_f99_s46, _s47` | ViT-L/16 full data, final.pt (epochs 8-11 full only) | yes | no |
| `lane_hex_20261108/deploy/` | cvw3 s46/s47, 4 detectors, 4 frozen heads, graph nets; `mtcps_c512_f99` (6.09 GB) full/t1 only | yes | graph nets |
| `green_track_20261028/deploy/` | hex6 bundle and 12 detectors (sg2pre, sg2, s8) | yes | yes |
| `lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46/raw` | attention-pool DINOv3-B (HEX column) | yes | no |
| `lane_cv_20261108/graph_models`, `frozen_heads_full` | 200 graph nets, 60 frozen heads | yes | yes |
| `lane_cv_20261108/arms/dinov3_b16_f99_s47, _s49` | CV 3-seed term | full only | no |
| `meta_ceiling_20261103/sweep/sim_pred` | simulation-embedding nets | yes | yes |
| `lane_cellroute_20261109/deploy_fp16` | 82 fp16 copies (22.31 GB) served instead of the originals; sha list `deploy_fp16_sha256.txt` | yes | yes |
| `phase2_backbone_screen_20261001/hf_cache`, `lane_gpu3_20261106/weights/dinov3-vitl16`, `phase2_foundation_20261002/weights` | pretrained base weights (rebuilds only) | - | - |

Closed-but-reusable weights (synthetic-pretrained members, order-loss DINOv3, window members, fold checkpoints) are listed per lane; the PI's
instruction is to keep models, so their checkpoints are not in any delete list until the PI says otherwise (section 4).

### 1.3 Upload files (build box `qudgks@210.107.192.239:~/workspace/clear-ec/package_20261112/upload_20261113/`)

| File | Bytes | sha256 (first 16) |
|---|---:|---|
| `clear-ec_20261112-freeze-9687076-p.tar.gz` (image dfc08a448e58) | 4,114,574,109 | 98f57ec5ac9fcfdc |
| `model_freeze-9687076-p_20261112.tar.gz` | 15,057,667,603 | fff3ec09e155841f |
| `clear-ec_20261112-freeze-9687076-t3.tar.gz` (image 8853d77c1f5a) | 4,114,563,350 | 4411cb7a7b376110 |
| `model_freeze-9687076-t3_20261112.tar.gz` | 8,809,046,691 | 7caaa02693776c7c |

Tested on the box as Grand Challenge runs it (RTX 4090, `--network none`, 16 GB, uid 10001): 17 of 17 jobs for each tier
[O-DEPLOY-022, O-DEPLOY-023]. Seven packaging bugs were found by the real-T4 and uid-10001 tests and fixed [F-DEPLOY-002/003/005].
`UPLOAD_TRYOUT_KO.md` in the same folder has the platform settings and the expected Try-out outputs.

## 2. Data

| What | Path | Notes |
|---|---|---|
| Official frames and labels | `/workspace/clear-ec-research/datasets/clear_ec_official/` | not regenerable |
| Whole-frame memmap | `datasets/clear_ec_fullframe_20260904/img_full_raw.dat` (8954 x 972 x 1296 uint8) | tests and training input |
| Five-fold split | `ssl_screen_20260911/preflight/fold{0,1}.npz`, `overnight_target_20260912/preflight/fold{2,3}.npz`, `target_fold4_20260912/preflight/fold4.npz` | defines every OOF; do not move |
| Full-data split (fold 99) | `fulldata_recipeA_20261028/fulldata_regression_split.npz` | 8,952 training rows; the 2 HEX-zero rows held |
| Labels, folds, donors, image statistics | `eval_harness_20261031/`, `meta_ceiling_20261103/source/best_pipeline_20261106.py` | imported by every lane |
| Best-pipeline inputs | `recipeA_analysis_20261028/hybrid_oof.{npz,json}`, `submission2_search_20261103/green_features.npz`, `meta_ceiling_20261103/sweep/` | Phase-1 OOF, recipe A OOF, LOFO scalars, submission 1 OOF |
| Per-metric OOF and parameters | `lane_cd_20261108/{reads,tta}`, `lane_cv_20261108/{out,deploy*,reads,graph_runs}`, `lane_hex_20261108/{preds,feats}`, `lane_cellroute_20261109/{out,inference/params,inference/manifests}` | strict OOFs, tier files, deploy params |
| Member predictions | `<lane>/arms|runs/.../predictions_*.npz` | regenerable only by retraining: keep even where checkpoints go |
| Segmentation features | `lane_hex_20261108/preds/seg_all.npz`, `lane_hex_20261108/seg_loose/c*.npz` | the CD mask slot reads `seg_loose` directly |
| Synthetic specular set v3 | `lane_backbone_20261108/data/synth/synth_v3_frames.u8` (63 GB) + labels | all synthetic routes closed; seeded generator in `source/synth_specular_20261111.py` |
| External | Topcon set (385 images, `datasets/external_*`), Vigueras-Guillen code and weights (GPL-3.0) | closed uses |

Further data sets found by the data analysis (`analysis_data_20261113.md`):

| What | Path | Notes |
|---|---|---|
| Raw download | `datasets/clear_ec_fullframe_20260904/train_mha/` (9,000 `.mha`), `datasets/clear_ec_official/final_train_ids.csv` (9,000 label rows, `slide_id` column) | all 972 x 1296 uint8, 9,000 unique IDs |
| Reconciled 9,000-row cache | `datasets/clear_ec_final9000_20260904/` | used only by the Phase-1 all-data fits (8,998 rows) |
| Pixel fingerprints | `ssl_screen_20260911/preflight/pixel_fingerprints.npz` | 14 pixel-identical pairs under different IDs |
| Expert cell markers | 25 frames (20 train the detectors, 5 validate; IDs in `green_point_probe_20261023/data_binding.json`) | the 20 are kept out of every stack fit (`good` mask, 8,934 rows) |
| External Topcon SREP set | 385 images with cell points, 307 used for the pre-only detectors | resampled x 1.296 |

**Row accounting (verified 11-13):**

| Step | Rows | Dropped |
|---|---:|---|
| Download: files and label rows | 9,000 | - |
| Cache and whole-frame memmap | 8,954 | 46 labelled, readable frames lost to an exact-string ID match (`code/cache_native.py:7`): trailing space 33, hyphen / space 10, other spelling 3; all new-format IDs |
| Scored CD, CV / HEX | 8,954 / 8,952 | two HEX-zero rows (`2019-0064-ODCN`, `-OSCN`) |
| Full-data fit, whole-frame models | 8,952 | the two HEX-zero rows; the 46 are NOT in these models |
| Full-data fit, Phase-1 crop members | 8,998 | the two HEX-zero rows; the 46 ARE in these models |
| Fold fits, whole-frame models | 7,150-7,161 per fold | 0-12 rows per fold linked to validation by pixel-identical frames |
| Stack and stretch fits | 8,934 | the 20 detector-training frames |
| Cell-route ridge fit | 8,825 | the 20 frames and 109 rows without usable features |

"8,953" in BRIEF and in the KG dataset node matches no file (prose error). IDs: 4,514 old-format rows (each its own donor group), 4,440
new-format rows in the cache (4,486 in the CSV); 6,768 donor groups; no group spans two folds. Preprocessing is CLAHE plus 1-99 percentile
scaling only (no line or edge enhancement); train-time augmentation of the whole-frame models is horizontal and vertical flips only.

## 3. Ideas

### 3.1 Adopted (in the frozen candidate)

| Metric | Idea | Evidence |
|---|---|---|
| CD | P1 / recipe A geometric reweighting; DINOv3-B/16 member; ViT-L/16 member and its seed mean; four-flip TTA over epochs 16-20; mask-density slot; strict nesting | -0.035; -0.047; stack -0.086, seed mean -0.0049; -0.008; -0.017 (all five folds) |
| CV | frozen token-statistic heads on the recipe A encoder; TTA-peak graph quartet; noise-weighted order-loss graph quartet; DINOv3-B term | heads +0.0224 if dropped; quartets +0.0118 strict if dropped; DINOv3 term +0.0078 strict if dropped |
| HEX | joint nested ridge instead of a sequential chain; zero-shot Cellpose-SAM segmentation features; attention-pool DINOv3-B column; CPSAM multi-target member (full/t1); CV-weighted DINOv3 seed mean; cell-route step | 9.7266 vs 9.7496; +0.0252 leave-one-out; -0.0069; +0.0012 if removed; +0.0031 if removed; -0.0064 end to end on t1 |
| All | checkpoint log-mean averaging; deploy fit = nested procedure with no outer fold (L5); strict reporting (L4) | L4 sizes: 0.006 (CV), 0.010 (HEX chain), 0.010 (cell-route) |
| Packaging | fp16 storage of regression members; skipped base weights; per-stage unloading (17.35 -> 4.33 GB GPU); tiers by build argument; T4 default fp16 + one detector copy; non-root readability check | parity median <= 0.009 %; T4-path impact +0.0066 (+0.002 to +0.012) on 200 rows |

### 3.2 Closed, with the number that closed it (selection; full lists in the lane files)

- **Synthetic pretraining**, all three routes and all three metrics: member-level gains (-0.072 DINOv3-B, -0.068 ViT-L, -0.006 recipe A) vanish in
  the stacks: CD -0.0031 / -0.0017 strict (inside seed noise), CV +0.0035 / +0.0041 / +0.0078 strict, HEX +0.0010 / +0.0017
  [D-SYN-001, D-SYN-006, D-SYN-008].
- **Other backbones and heads**: ConvNeXt-B (weight 0.05, ~0), ViT-H+/16 (-0.0034 two folds, cost), DINOv3 stride 8 (member 6.88 vs 6.58),
  distribution head (+0.008), attention pooling as a CD/CV member (+0.004 / weight 0), CPSAM 512 as a CD member (+0.023).
- **Training variants**: C-Mixup (+0.0099 CD, +0.0073 CV, +0.0153 HEX), drop path 0.2 (+0.027), ViT-L drop path 0.3 + layer-wise decay (stopped,
  ~+1.1 at epoch 5), geometric augmentation (+0.116), auxiliary detector targets (+0.019), target denoising (+0.146), row reweighting (-0.0025),
  distillation (44 % of the gap), CD-only loss (1e-5: -0.0030, 3e-5: +0.0181 on two folds), encoder LR x5 (validation +0.33, pre-lanes).
- **Views**: technician-window crops (CD +0.0026), native tiles (CD +0.255 member), shading correction (worse).
- **Label-free and graph CD columns**: Fourier ring (weight 0), detector density (weight 0), cell-graph CD column (0 to +0.010).
- **Post-hoc**: global / per-level / isotonic recalibration (+0.004 to +0.050); blending submission 1 (+0.0024) or the Phase I best model
  (+0.0025; sixth member -0.0005 n.s.) [O-CD-SUB1-CALIB, O-CD-PHASE1-CALIB]; from-scratch combination searches in all three lanes (no gain).
- **CV**: order-loss DINOv3, ViT-L / ViT-H members, weighted loss, graph variants (+0.005 to +0.116), segmentation features (+0.011 to +0.020).
- **HEX**: Vigueras-Guillen edge network (+0.0003), forward selection from scratch (worse), segmentation variants, HEX-weighted CPSAM loss.
- **Deployment numerics**: all-fp32 on T4 (+0.0069 vs fp16 +0.0066, and > 600 s), mixed fp32 (365 s with hex6 only), A10G proxy (+0.0059),
  0.05 % parity off the H200 (unattainable) [O-COORD-F32-200, O-DEPLOY-020].
- **Test-gap adjustment**: submission 1's pattern is inside 1,000-image sampling noise; an offset has negative expected value [F-TESTGAP-001].
- **Source shift**: leave-one-source-out costs CD +3.4/+3.7, CV +6.0/+6.5, HEX +3.5/+0.8 points, 85-90 % a level offset [F-SHIFT-001].

### 3.3 Open or only partly tested

| Idea | Status |
|---|---|
| Recipe A hyperparameters never varied: layer-wise decay, weight decay, warm-up, batch size, joint-loss weights, input resolution, > 20 epochs, appearance augmentation | no run; `--appear` exists in the runner |
| Attention-pool column seed audit | adopted column rests on seed 46 only |
| HEX deploy base convention | chain convention 9.7266 vs OOF-input base 9.7414; effect on test unmeasured |
| Test HEX label-mix shift | the one covariate shift that fits the test pattern; unobservable without test labels |
| CD label scatter within donors; six impossible-label frames (0.187 MAPE) | measured, no action possible without relabelling |
| Quartet-free CV stack sharing detector flips with HEX | would save up to 3 of 4 detector flips for +0.0118 |
| p model archive under 10 GB | move 5.5-6.3 GB into the image or more fp16 copies; untested |
| Synthetic recipe A seed mean on folds 1-4; full-data synthetic builds | not run (routes closed) |

## 4. Clean-up plan

Rules used: models are kept (PI); nothing a rebuild of p / t3 needs is touched before the submission is final; member prediction files are
kept wherever checkpoints go; by-name references are checked with grep before any folder is deleted.

**Must not move (rebuild closure, deploy lane section 5):** `lane_cellroute_20261109`, `lane_cd_20261108`, `lane_cv_20261108`,
`lane_hex_20261108`, `lane_backbone_20261108`, `lane_gpu2_20261106`, `green_track_20261028`, `meta_ceiling_20261103`, `eval_harness_20261031`,
`oof_recipeA_20261027`, `phase2_crossfold_20261015`, `phase2_foundation_20261002`, `submission1_test_20261030`, `fulldata_recipeA_20261028`,
`deploy_review_20260905`, `recipeA_analysis_20261028`, `submission2_search_20261103`, the split folders of section 2, `cpsam_models/`.
Hard-coded paths: every lane's `source/*.py` constants, `inference/members.py:142`, `params_build.py`, the manifests, `det_peaks_image.py`.

| Class | Items |
|---|---|
| DELETE-NOW (scratch, partial, stale; owners' call) | CD: `archive/{failed_*,stopped_*,_smoke*,_check_vitl_*}`, vith16 partial, `arms/_archive_cvlane/`, `queue*.txt`, `STOP*`. CV: `graph_models/_archive_wrong_seed_20261110/`, `arms/_failed_nan_*`, `arms/_archive/`, `locks/`. HEX: `runs/vig_maps/`, `seg_rot/`, `tmp_pseudo_test*`, `tmp_gnn_check_*.npz`, `pylib/`. Backbone: suffix-less queue-marker folders in `runs/`, `runs/*.lock|*.pid`, `HOLD_*`, `STOP_*`, `tmp/`, `archive_holds_20261111/`, `data/synth/_archive_v1*`, `_archive_v2*`. Build box: `upload_20261113/stale_perm600_20261113/`, `package_20261112/t1/context`, dangling image b982cbddf583. Pod `/root`: Colab ssh config entries, `/root/bin/cloudflared`, Colab CLI with the PI's saved login (stop pid 599163 first) |
| DELETE-AFTER-SUBMISSION | fold and closed-arm checkpoints in every lane (predictions kept) **unless the PI wants every checkpoint kept**; `lane_backbone_20261108/data/synth/synth_v3_frames.u8` (63 GB); `lane_cv_20261108/{graphs,frozen_own_*,frozen_dino_*,deploy_fallback_10.1101}`; recipe A epochs 1-7 and last.pt; `package_20261112/{model,t1/model,t2/model}` (hard links, frees nothing); build box `{p,t3}/{context,model}`, `test_model_*`, `test_20261113`; `/root/.ssh/buildbox_20261112*` and its line in the box's `authorized_keys` |
| ARCHIVE (small evidence) | every lane's `logs/`, `RESULTS.tsv`, `REPORT.md`, `reads/`; `integration_rehearsal_20261110`; `rootcause_20261109`; `error_case_analysis_20261106`; Phase I legacy (`raw_ssl_next_20260913`, `region_candidate_20260907`, `deploy_methods_20260905`); Colab scripts; Vigueras files |
| No owner yet (needs a reference check before any call) | `batchcheck`, `cd_density_channel_20261106`, `context_20260904`, `crazy_hypsel_20261110`, `crop_dino_screen_20260910`, `crop_h200`, `curriculum_hpo`, `deploy_region_20260907`, `deploy_shift_20261107`, `domain_jepa_*`, `final_candidate_20260904`, `hybrid_20260905`, `lane_gpu0/1_20261106`, `methods_20260904_smoke`, `oof_h200`, the remaining `phase2_*`, `queue_runner_20261027`, `recipe_A_20261027`, `region_*`, `smoke`, `specialize_screen_20260908`, `submission2_test_20261101`, `t4_dry_run_*`, `target_screen_20260912`, top-level `submission*`, `/workspace/{artifacts,checkpoints,logs,scratch_*}` (`scratch_hex6_bundle` is not in the deploy closure list; confirm by grep) |

Conflicts between lanes, resolved: `lane_hex_20261108/seg_loose` stays (the CD stack reads it; the HEX lane had marked it deletable);
`submission1_test_20261030` and `lane_gpu2_20261106` stay (in the rebuild closure; the CV lane had marked the first as archive).

## 5. Loose ends

- **Unverified:** `cellroute_deploy_{t2,t3}.json` against a no-outer-fold refit (full / t1 was checked by the lanes; the p file's kept-version fit
  was recomputed on 11-13 by the post-processing analysis and reproduces to 2.8e-17, its version choice was not rerun); the platform T4 exec time of p;
  the 10 GB upload cap; the attention-pool column's second seed; which CV parameter folder the box t3 test read (deploy lane is checking).
- **Corrections on record:** O-BB-SYNFT3-5F (8.6736 was the 11-10 rehearsal OOF, not the frozen candidate: 8.66336 / 8.66867 strict)
  [O-BB-SYNFT3-5F-CORR]; A-DEPLOY-010 superseded by A-DEPLOY-011; image b982cbddf583 superseded; every pod and Colab test ran as root and
  covered no file permissions; "T4 not viable" and "A10G required" superseded by D-PKG-006.
- **Stale metadata:** manifests deploy / deploy_p / deploy_t3 still list `lane_cv_20261108/arms/dinov3_order_f99_s46` (unused).
- **Strict finding:** the full candidate's extra ~9 GB of CD weights buys nothing measurable over CD t1 (6.1743 vs 6.1737).
- **Records behind the ledger:** the lanes' `RESULTS.tsv` files lack the 11-13 reads; BRIEF and the KG lack 11-12 and 11-13 (tiers, T4, packaging
  bugs, fp32 read, calibration reads). The deadline in BRIEF (2026-11-15 06:00 Toronto) was not reconciled with earlier notes (11-14).

## 6. Analysis reports written after this summary (same notes folder)

`analysis_hyperparameters_20261113.md`, `analysis_data_20261113.md`, `analysis_postprocessing_20261113.md`, `analysis_bias_20261113.md`,
`analysis_cd_literature_20261113.md` (each by a separate agent, read-only; in progress when this file was written).
