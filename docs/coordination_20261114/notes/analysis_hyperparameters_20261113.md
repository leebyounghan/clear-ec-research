# Training hyperparameters of the tier p candidate (analysis, 2026-11-13)

Scope: every trained model that the frozen submission candidate "tier p" loads, as listed in
`experiments/lane_cellroute_20261109/inference/manifests/deploy_p.json`.
Read-only analysis. Nothing was created, changed or started on the pod.

## Conventions

- Paths without a prefix are relative to `/workspace/clear-ec-research/experiments/` on pod `clear-ec-research-4gpu-1104`.
  `code/...` is the local repo `clear-ec-research/code/` (sha256 of the files matches the hashes stored in the checkpoints, see 1.1).
- "config" = a value stored by the run itself (manifest, launch.json, result.json, checkpoint config). "script" = a constant in the
  training script, with no separate run record. "summary" = taken from a lane summary or KG note, not re-read from the run.
- Schedule name "schedule10": linear warm-up over epoch 1, cosine from 1.0 to 0.1 of the base LR until the end of epoch 10, then flat at 0.1
  (`fulldata_recipeA_20261028/regression_std_s46/source/phase2_backbone_ssl_regression_20261020.py:390-394`).
- "Recipe loss": `0.5 * mean|z - zt| + 2.5 * mean|expm1((z - zt) * sigma)|`, z = standardised log prediction, zt = (log y - mu) / sigma with
  mu, sigma from the training rows; the second term equals 0.5 x MAPE / 0.2; the three metrics have equal weight
  (`.../source/phase2_foundation_train_20261002.py:159-161`).
- "SSL5": 5 epochs of LeJEPA self-supervised adaptation of the encoder before regression (details in 1.2).
- All scores are MAPE in %. Every number in section 2 names its baseline and folds as the source gives them.
- No AdamW betas or eps are set in any script I read; the configs do not record them (library defaults apply).

## 0. What tier p loads (from the manifest)

| Manifest key | Files | Family below |
|---|---|---|
| `p1_members`, `p1_cpsam` | 5 crop CPSAM members + `cpsam_models/cpsam_v2` | 1.1, 1.8 |
| `recipeA_tta`, `cv_parts.token_encoder` | recipe A fold99 s46, s47 `epoch_20.pt` | 1.2 |
| `dino_cd`, `dino_cv` | DINOv3-B/16 fold99 s46b, s47 `epoch_16..20.pt` | 1.3 |
| `vitl_seeds` | ViT-L/16 fold99 s46, s47 raw `final.pt` | 1.4 |
| `cvw3` | CV-weighted DINOv3-B fold99 s46, s47 `epoch_08..12.pt` | 1.5a |
| `detectors_all16`, `preonly_detectors`, `hex6_deploy_dir` | 12 + 4 detectors, 4 frozen heads, hex6 parameters | 1.5b, 1.5c |
| `hex_graph_nets`, `cv_parts.graph_models` | 125 + 200 graph networks | 1.5d, 1.7a |
| `hex_extra.attnp5` | attention-pool DINOv3-B fold99 s46 raw `epoch_08..11.pt`, `final.pt` | 1.6 |
| `cv_parts.frozen_heads` | 60 frozen token-statistic heads | 1.7b |
| (CV/HEX ridge input) | `meta_ceiling_20261103/sweep/sim_pred/sim2_l8_s{0,1,2}.pt` | 1.7c |
| `cpsam_seg` | `cpsam_models/cpsam_v2` (segmentation) | 1.8 |

`orderdino_cv` (order-loss DINOv3-B, `lane_cv_20261108/arms/dinov3_order_f99_s46`) is listed in the manifest but, per the CV lane summary,
no tier's components use it and the p census holds no file of it. It is not covered below.
The manifest's `vitl_note` ("not built yet; paths are a guess") is stale: both `final.pt` files exist and have `complete.json`.

## 1. Hyperparameters per family

### 1.1 Phase-1 crop CPSAM members (five files in `deploy_review_20260905/staged/model`)

Values common to the five members:

| Item | Value | Source |
|---|---|---|
| Backbone, initial weights | Cellpose-SAM encoder (`base.net.encoder` of `CellposeModel(pretrained_model='cpsam_models/cpsam_v2')`), 256-channel output | `code/final_candidate_core.py:72-76` |
| Input | Central 512x512 of the frame (`img_native_512.dat`); training view = one random 256x256 crop, random H flip, V flip, rot90; CLAHE(2.0, 8x8) + 1-99 percentile scaling per tile; grey copied to 3 channels | `code/final_candidate_core.py:13,21-24,48-54`; `code/train_final_candidate.py:36` |
| What is trained | Full fine-tune: encoder and heads, gradient checkpointing per block; no layer-wise decay | `code/train_final_candidate.py:38-44` |
| Optimizer | AdamW, two groups | `code/train_final_candidate.py:42`; `code/train_deploy_methods.py:103-105` |
| Learning rates | Encoder 1e-5, heads 1e-3 | same lines |
| Weight decay | 1e-4, all parameters | same lines |
| Gradient clip | Global norm 1.0 | `code/train_final_candidate.py:67`; `code/train_deploy_methods.py:182` |
| Batch, accumulation | 48, no accumulation | checkpoint `config.batch_size`; `code/train_final_candidate.py:46` |
| Head, pooling | Global mean pool -> LayerNorm(256) -> three separate heads, each Linear(256,128)-GELU-Dropout(0.2)-Linear(128,1), last layer zero-init | `code/final_candidate_core.py:57-65` |
| Drop path | None in the scripts | `code/train_*.py` (no such option) |
| Target transform | Standardised log with fixed constants LM = [7.9199, -0.9825, -0.6211], LS = [0.1897, 0.1881, 0.2749] | `code/final_candidate_core.py:10-11`; checkpoint `config.lm`, `config.ls` |
| Mixed precision | bf16 autocast, TF32 allowed | `code/train_final_candidate.py:32,64` |
| EMA / SWA | None | scripts |
| Training rows | 8,998 (all-data fit, no validation rows; metric35 and predbag2_35 record 2 excluded non-positive HEX rows) | checkpoint `config.train_rows`, `excluded_nonpositive_hex` |
| Inference | 5 crops (four 256 quadrants + centre of the central 512) x 8 dihedral views = 40 views, mean of the standardised outputs, then exp; bf16, view batch 8 | `code/final_candidate_core.py:12,37-45,92-99` |

Per member (all values from the `config` stored inside each `.pt`):

| Member | Seed | Loss | Schedule | Checkpoint used | Train time |
|---|---|---|---|---|---|
| seed42, seed77, seed2026 | 42 / 77 / 2026 | L1 on the standardised log (`train_final_candidate.py:64`) | Linear warm-up 200 updates, then constant LR; 25 epochs = 4,700 updates run (`:45-46`) | Update 3,576 (epoch 20) / 2,831 (16) / 2,980 (16) = median over the five folds of the OOF-best update count, fixed before the all-data fit (`final_candidate_20260904/plan.json`) | 6,584 / 6,608 / 6,596 s for the full 25 epochs (`final_candidate_20260904/runs/seed*/result.json`) |
| metric35 | 42 | "mixed_mape": 0.5 x log-L1 + 0.5 x relative error / 0.2 (`method_training_core.py:19-31`) | Cosine with floor 0.05, warm-up 200 updates, length 35 x 149 = 5,215 updates (`train_deploy_methods.py:23`, `method_training_core.py:9-16`) | Update 4,470 (epoch 24) | 6,232 s (`deploy_methods_20260905/runs/metric35/result.json`) |
| predbag2_35 | 42 | "log_l1"; the model output is the mean over 2 random crops per image (`k_crops` 2, `train_deploy_methods.py:30-36,44-47`) | Same cosine as metric35 | Update 3,874 (epoch 21) | 11,604 s (`deploy_review_20260905/runs/predbag2_35/result.json`) |

Notes on provenance:
- `code/train_final_candidate.py` hashes to 09239f88..., the hash stored in the seed members. `code/train_deploy_methods.py` hashes to
  8cafb7ca..., the hash stored in predbag2_35. metric35 stores a different hash for that file (3c7135c3...); that revision was not found,
  so its row rests on its checkpoint config and on `RECIPES` in the current file.
- The seed members store `final_candidate_core.py` hash 08707772...; the current and staged file hashes to 92207fff....
- Folds: the five-fold OOF runs are in `followup_20260902`, `submission_20260904`, `methods_20260904` (CD lane summary). Their run
  configs were not opened; the only fold facts read are the per-fold best epochs in `final_candidate_20260904/plan.json`
  (train rows 7,161-7,163 per fold, batch 48).
- In tier p the five members are calibrated with the OOF SCALE of the CD lane, not with `candidate5.json` (manifest `p1_note`).

### 1.2 Recipe A = CPDINO ViT-B whole frame (`fulldata_recipeA_20261028/regression_std_s46`, `_s47`)

SSL stage (LeJEPA, "SSL5"), full-data run `fulldata_recipeA_20261028/ssl5_s1003`:

| Item | Value | Source |
|---|---|---|
| Backbone, initial weights | CellposeDINO ViT-B (`CPDINO(model_name='vitb', ps=8)`), patch kernel 16, stride 8, padding 4; weights `phase2_foundation_20261002/weights/cellpose/cpdino-vitb`, revision 7c61431b..., sha256 3ed4c06a...; 85,669,632 encoder parameters | `ssl5_s1003/run/result.json` (`details`); `regression_std_s46/source/phase2_foundation_preflight_20261002.py:112-141` |
| Input | Whole native 972x1296 frame, replicate-pad bottom 20 / right 16 to 992x1312; two views per image | `ssl5_s1003/manifest.json` (`recipe.view`, `views`) |
| Objective | LeJEPA: (1 - 0.02) x invariance + 0.02 x SIGReg (256 directions, 17 knots); projector 768-2048-BN-ReLU-2048-BN-ReLU-128 (6,041,728 parameters, training only) | manifest `recipe.lamb`, `sigreg_directions`, `knots`; `ssl5_s1003/source/phase2_crossfold_ssl_20261015.py:292-295`; `.../phase2_lejepa_20261003.py:172-174` |
| Augmentation | Independent H/V flips, gain [0.9, 1.1], offset [-0.03, 0.03]; no resize, crop or mask | manifest `recipe.augmentation` |
| Optimizer, LR, WD | AdamW; encoder 1e-5, projector 1e-3; weight decay 1e-4 | manifest `recipe`; `phase2_crossfold_ssl_20261015.py:296-298` |
| Schedule | Linear warm-up over 20 updates, then fixed LR | manifest `recipe.lr_policy`; `:521-523` |
| Epochs, updates, batch | 5 epochs, 1,400 updates, 32 images per update (x 2 views), micro-batch 4 views (gradient caching) | manifest `recipe`, `total_steps` |
| Clip, precision | Clip 1.0; encoder bf16, projector and loss fp32 | manifest `recipe` |
| Pooling in SSL | Mean of patch tokens | manifest `recipe.pooling` |
| Seed | 1003 | manifest |
| Rows | Full-data: all 8,954 images, no labels. Fold runs: the fold's training rows only | `alldata_ssl_addendum.json`; `oof_recipeA_20261027/protocol.json` |
| Time | 21,249 s | `ssl5_s1003/run/result.json` (`seconds`) |

Regression stage (values from `regression_std_s46/manifest.json` block `recipe` unless another source is named):

| Item | Value | Source |
|---|---|---|
| Initial weights | Encoder from `ssl5_s1003/run/last.pt` (sha 0f3c1572...); projector and SSL optimizer not loaded | manifest; `smoke.json` |
| Input | Whole native 972x1296, replicate-pad to 992x1312; token grid 124x164 (kernel 16, stride 8); CLAHE(2.0, tile grid 40x30) + 1-99 percentile scaling to [0,1]; no ImageNet normalisation for CPDINO | manifest `recipe.view`, `normalization`; `source/phase2_foundation_train_20261002.py:127-129,144-146`; `smoke.json` |
| What is trained | Full fine-tune, all 12 blocks and the head (85,769,987 trainable parameters); no layer-wise decay, no frozen part | `smoke.json` (`all_12_encoder_blocks_updated`, `trainable_parameter_count`) |
| Optimizer | AdamW (fused), two groups | `source/phase2_backbone_ssl_regression_20261020.py:341-343` |
| Learning rates | Encoder 1e-5, head 1e-3 | manifest `recipe.encoder_lr`, `head_lr` |
| Schedule, warm-up | schedule10: warm-up 1 epoch (560 updates), cosine to 0.1 by the end of epoch 10, flat after. `history.json` shows encoder LR 1.27e-6 at the end of epoch 9 and 1.0e-6 at epochs 10, 11 and 20 (head LR 1.0e-4 at epoch 20) | manifest `warmup_epochs`, `schedule_epochs`, `minimum_lr_ratio`; `:390-394`; `runs/fold99/cpdino_vitb/history.json` |
| Epochs, updates | 20 epochs, 11,200 updates | manifest; `progress.json` |
| Batch, accumulation | Effective 16 = 2 GPUs (DDP) x micro-batch 4 x 2 accumulation rounds | manifest `execution` |
| Weight decay | 1e-4, all parameters | manifest `recipe.weight_decay` |
| Gradient clip | Global norm 1.0. Clipped fraction of updates: 0.891 (s46) and 0.896 (s47) in epoch 1; 1.00 in epochs 5, 10 and 20 for both seeds | manifest; `history.json` (`gradient_clip_fraction`) |
| Drop path / dropout | None (no such module in the build or the head) | `source/phase2_foundation_preflight_20261002.py:112-141,189` |
| Loss, target | Recipe loss; standardised log, train-only mean [7.9206, -0.9806, -0.6180] and std [0.1870, 0.1872, 0.1602]; equal metric weights | manifest `recipe.objective`, `log_mean`, `log_std` |
| Pooling, head | Population std of the final normed patch tokens (768) -> LayerNorm -> Linear(768,128) -> GELU -> Linear(128,3); last layer N(0, 1e-3), zero bias (100,355 parameters) | manifest `recipe.head`, `pooling`; `source/phase2_pooling_20261019.py:13-30`; `source/phase2_foundation_train_20261002.py:176` |
| Augmentation | Independent H/V flips, deterministic in (seed, epoch, image index) | manifest `recipe.augmentation` |
| Mixed precision | bf16 autocast | manifest `recipe.precision` |
| EMA / SWA | None; weights of every epoch are saved | manifest `diagnostics.epoch_weights` |
| Seeds | 46 and 47 in tier p (a seed 48 full-data run also exists and is in no manifest) | folder listing; `fulldata_regression_addendum.json` |
| Folds | Folds 0-4 for seeds 46 and 47 (seed 48 on folds 0 and 4 only; cut on folds 1-3); fold 99 = 8,952 training rows, the two HEX-zero rows held as a placeholder | `oof_recipeA_20261027/protocol.json`; backbone summary section 6; `fulldata_regression_addendum.json` |
| Used at inference (tier p) | `epoch_20.pt` of both seeds, four-flip TTA, arithmetic mean over seeds, times per-metric scalars. The CD "t1" form uses the same epoch-20 TTA only. The s46 epoch-20 encoder is also the token encoder of the CV frozen heads and the initial encoder of every detector | `lane_cellroute_20261109/inference/members.py:97-99`; manifest `recipeA_tta`, `cv_parts.token_encoder`; CD summary tier table |
| Training time | 16,937 s (s46), 16,867 s (s47), sum of `epoch_seconds`, on 2 x H200; peak 35.2 GiB per GPU | `history.json` |

Launch command of the s47 run: `queue_lane01/fulldata_s47_train.log:1`
(`--pooling std --seed 47 --ssl-epochs 5 --head-lr 0.001 --clip 1 --epochs 20 --schedule-epochs 10`).

### 1.3 DINOv3-B/16 members (`lane_cd_20261108/arms/dinov3_b16_f99_s46b`, `_s47`)

| Item | Value | Source |
|---|---|---|
| Backbone, initial weights | DINOv3 ViT-B/16 (HF `AutoModel`, `phase2_foundation_20261002/weights/dinov3-vitb16`), patch stride 16; then SSL5 with the recipe of 1.2 (seed 1003, 5 epochs, 1,400 updates, batch 32, encoder LR 1e-5, mean pool), encoder swapped | `lane_cd_20261108/source/family_cd_20261108.py:54,62,82-97,205-228`; `arms/dinov3_b16_f99_s46/ssl/complete.json` |
| SSL reuse | s46b and s47 both start from `arms/dinov3_b16_f99_s46/ssl/last.pt` (the s47 member has no SSL of its own) | `arms/dinov3_b16_f99_s46b/launch.json`, `arms/dinov3_b16_f99_s47/launch.json` |
| Input | Whole frame, replicate-pad to 992x1312, grid 62x82; CLAHE + percentile as recipe A, then ImageNet mean/std | `family_cd_20261108.py:59,62,174-182` |
| What is trained | Full fine-tune, no layer-wise decay | `:282` |
| Optimizer, LR, WD | AdamW (fused); encoder 1e-5, head 1e-3; weight decay 1e-4 (script constants) | `:282,295` |
| Schedule | schedule10 (`schedule_epochs` 10) | `launch.json`; `:294` |
| Epochs | 20 | `launch.json`, `runs/fold99/dinov3_b16/result.json` (`recipe`) |
| Batch, accumulation | 16 per update, micro-batch 16 (no accumulation), one GPU | `launch.json` (`micro`); `:291,297` |
| Gradient clip | 1.0 | `:308` |
| Drop path / dropout | None (the runner has no such option) | `family_cd_20261108.py` |
| Loss, target | Recipe loss (`cd_loss: "recipe"`); standardised log from the training rows | `launch.json`; `:44-46,283` |
| Pooling, head | std pool -> LayerNorm -> Linear(768,128) -> GELU -> Linear(128,3), last layer N(0, 1e-3) | `:262-264` |
| Mixed precision | bf16 | `:299` |
| EMA / SWA | None; weights of epochs 16-20 kept | `:331-332` |
| Seeds | 46 (run "s46b"; the first s46 run is logged with rc 1 after its SSL, `queue_done.txt:36`) and 47 | `launch.json`; `queue_done.txt:36,46` |
| Folds | Folds 0 and 4: `lane_gpu3_20261106/runs/dinov3_b16_f{0,4}_s46`, `lane_gpu2_20261106/runs/dinov3_b16_f{0,4}_s47`; folds 1-3: `lane_cd_20261108/arms/dinov3_b16_f{1,2,3}_s{46,47}`; same launch values (checked f1 s46, f3 s47, f0 s46, f4 s47) | the four `launch.json` files |
| Used at inference (tier p) | CD: both seeds, epochs 16-20, four-flip TTA, mean in log space. CV: s46b only, epochs 16-20, single view | `members.py:112-122`; manifest `dino_cd`, `dino_cv` |
| Training time | Regression 4,071 s (s46b), 4,074 s (s47); SSL 2,703 s | `result.json` (`seconds`); `ssl/complete.json` |

### 1.4 ViT-L/16 DINOv3 members (`lane_cd_20261108/arms/vitl16_f99_s46`, `_s47`)

All values from `launch.json` of the two runs (identical except seed and GPU) unless another source is named.

| Item | Value | Source |
|---|---|---|
| Backbone, initial weights | timm `vit_large_patch16_dinov3.lvd1689m` (`hf_cache` snapshot 30c11095...), stride 16, `dynamic_img_size`; then SSL5 (family `dinov3_l16t`); s47 reuses the s46 SSL | `lane_cd_20261108/source/family_bb_cd_20261108.py:40,51,58-69`; `launch.json` (`ssl`); `queue.txt:94,96` |
| Input | Whole frame padded to 992x1312, grid 62x82, ImageNet normalisation | `family_bb_cd_20261108.py:51` |
| What is trained | Full fine-tune; layer-wise decay off (`llrd` 0.0; all 26 encoder groups at 1e-5) | `launch.json` (`llrd`, `groups`) |
| Optimizer, LR, WD | AdamW (fused); encoder 1e-5, head 1e-3; weight decay 1e-4 | `launch.json` (`enc_lr`, `head_lr`, `wd`); `bb_runner_cd_20261108.py:213` |
| Schedule | schedule10; with 12 epochs the last two run at 0.1 of the base LR | `launch.json` (`schedule_epochs` 10, `epochs` 12) |
| Epochs | 12 | `launch.json` |
| Batch, accumulation | 16 per update, micro-batch 8 (2 accumulated) | `launch.json` (`micro`); `bb_runner_cd_20261108.py:241,255` |
| Gradient clip | 1.0 | `bb_runner_cd_20261108.py:272` |
| Drop path / dropout | Drop path 0.0 (off) | `launch.json` (`drop_path`, `drop_path_rates: []`) |
| Loss, target, head | Recipe loss; std pool head as recipe A (dimension 1,024) | `bb_runner_cd_20261108.py:259`; `family_bb_cd_20261108.py` |
| Augmentation | H/V flips only (`transpose_p` 0, `zoom` 1.0, no auxiliary targets, no teacher) | `launch.json` |
| Mixed precision | bf16 | `bb_runner_cd_20261108.py:257` |
| EMA / SWA | Three averaged tracks are computed and saved: EMA 0.998 and 0.999 from epoch 3, uniform SWA from epoch 6. Tier p loads none of them; it loads the raw track | `launch.json` (`ema`, `ema_start`, `swa_start`, `tracks`); manifest `vitl_seeds` |
| Seeds | 46 and 47 | `launch.json` |
| Folds | s46: folds 0 and 4 in `lane_backbone_20261108/runs/vitl16_avg_f{0,4}_s46` (EMA 0.999 / 0.9995 there, `keep_epochs` empty), folds 1-3 in `lane_cd_20261108/arms/vitl16_f{1,2,3}_s46`; s47: `arms/vitl16_f{0..4}_s47`; other values equal (checked f0 s46, f1 s46, f0 s47) | the three `launch.json` files |
| Used at inference (tier p) | raw `final.pt` (epoch 12), four-flip TTA, mean over the two seeds in log space | `members.py:125-128` |
| Training time | Regression 6,844 s (s46, GPU 0), 15,083 s (s47, GPU 3); SSL 7,475 s | `exit.json`; `arms/vitl16_f99_s46/ssl/complete.json` |

### 1.5 HEX lane models

#### 1.5a CV-weighted DINOv3-B, "cvw3" (`lane_hex_20261108/deploy/cvw3_f99_s46`, `_s47`)

Same runner and values as 1.3 except the rows below.

| Item | Value | Source |
|---|---|---|
| Loss | Recipe loss with per-metric weights [1, 3, 1] / 5 x 3 = [0.6, 1.8, 0.6] on (CD, CV, HEX), i.e. CV x 3 | `lane_hex_20261108/source/family_cvw3_copy.py:90-92`; `launch.json` (`cd_loss: "cvw3"`) |
| Epochs | 12 (schedule10) | `launch.json`; `source/gpu1_deploy_queue.sh:18` |
| Initial weights | DINOv3-B/16 + the CD lane's full-data SSL (`lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt`), both seeds | `launch.json` (`ssl`) |
| Weights kept | Epochs 8-12 (the runner keeps the last five) | `family_cvw3_copy.py:373-374` |
| Seeds, folds | 46 and 47; fold runs s46 in `lane_cv_20261108/arms/dinov3_cvw3_f{0..4}_s46`, s47 in `lane_hex_20261108/runs/cvw3_f{0..4}_s47` (fold configs not opened) | `launch.json`; HEX and CV summaries |
| Used at inference | Epochs 8-12 of each seed, single view, log mean; the ridge reads the HEX column | `members.py:177-180`; manifest `cvw3` |
| Training time | 2,492 s (s46), 2,449 s (s47) | `runs/fold99/dinov3_b16/result.json` |

#### 1.5b Point detectors with a fine-tuned encoder (12 in `green_track_20261028/deploy/detectors`, 4 in `lane_hex_20261108/deploy/detectors`)

| Item | Value | Source |
|---|---|---|
| Backbone, initial weights | CPDINO ViT-B encoder of full-data recipe A s46 `epoch_20.pt` (sha 1b86c328...) | each detector's `.json` (`init_encoder`, `init_sha`) |
| Head kinds | `s8`: Linear(768,1) on the last tokens, stride-8 map 124x164. `s8multi`: Linear(2304,1) on blocks 5/8/11. `sg2`: Linear(768,16) + 4x4 pixel shuffle, stride-2 map 496x656 | `green_track_20261028/source/green_point_features_20261027.py:32,35,99-104`; `green_sg2_specialist_20261028.py:43-56` |
| Training data | 20 "train-green" annotated frames x 4 exact flips; ROI-masked Gaussian targets, sigma 8 px (3.5 px and 5 px for `sg2_sigma3p5`, `sg2_sigma5`) | `green_deploy_train_20261030.py:1-6`; `green_point_spatial_20261023.py:7`; `lane_hex_20261108/source/deploy_det.py` |
| What is trained | Encoder and head | `green_deploy_train_20261030.py:69` |
| Optimizer, LR, WD | AdamW; encoder 1e-5, head 1e-3; weight decay 1e-4; constant LR (no scheduler in the loop) | `:69-86` |
| Updates, batch | 1,000 updates, batch 2 | `:29`; each `.json` (`updates`, `batch`) |
| Loss | Binary cross-entropy with logits inside the ROI | `:83` |
| Clip, precision | Clip 1.0; encoder forward in bf16 | `:77,85` |
| Seeds | `s8` 46, 47, 48; `s8multi` 47; `sg2` 46-50 (green track) and 51, 52 (HEX lane); `sg2_sigma3p5` 46; `sg2_sigma5` 46 | file names and `.json` files |
| Pre-only detectors `sg2pre_s46/47/48` | Same encoder and `sg2` head, trained only on 307 external SREP images rescaled x 1.296, hull-masked Gaussian BCE, 2,000 updates, batch 2, same optimizer; never fine-tuned on CLEAR-EC frames | `green_track_20261028/deploy/detectors/sg2pre_s46.json`, `sg2pre_s47.json`; `green_external_srep_20261030.py:213-270`; `queue_gpu3/ext_preonly_s47_pretrain_fold0.log:1` |
| Inference | Peak threshold 0.35 (fixed; chosen on the 20 train-green frames by the original specialist CV) | `sg2pre_s46.json` (`threshold_origin`) |
| Folds | Fold versions start from each fold's recipe A encoder (green track); the deployed files start from the full-data encoder | `green_deploy_train_20261030.py:1-6` |
| Training time | 379-381 s per detector (`s8_s46` 381.4, `sg2_s46` 378.7, `sg2_s51` 379.3); `sg2pre_s46` 279.7 s | the `.json` files (`seconds`) |

#### 1.5c Frozen-encoder detector heads (`lane_hex_20261108/deploy/frozen_heads`) and hex6

| Item | Value | Source |
|---|---|---|
| Encoder | Full-data recipe A s46 `epoch_20.pt`, frozen | `lane_hex_20261108/source/deploy_frozen.py:9-13`; `result.json` (`frozen_encoder: true`) |
| Head | `sg2` head (Linear(768,16) + pixel shuffle) | `green_track_20261028/source/green_frozen_heads_20261031.py:34` |
| Optimizer | AdamW on the head only, LR 1e-3, weight decay 1e-4, constant | `:35` |
| Updates, batch | `sg2f` (20 train-green frames): 3,000 updates; `sg2pref` (SREP only): 4,000 updates; batch 2 | `deploy/frozen_heads/sg2f_s46/fold0/result.json`, `sg2pref_s46/fold0/result.json`; `:49-53` |
| Seeds | `sg2f` 46, 47, 48; `sg2pref` 46 | folder names |
| Training time | `sg2f_s46` 189.5 s; `sg2pref_s46` not read | `sg2f_s46/fold0/result.json` |
| hex6 correction | No trained weights of its own: ten detectors (3 `s8`, 5 `sg2`, `s8multi_s47`, `sg2pre_s46`) and a fitted blend a = -0.2084, b = 0.5989, w = 0.28 on 8,925 rows | `green_track_20261028/deploy/ACTIVE.json`, `hex6_blend_params.json` |

#### 1.5d HEX graph networks (`lane_hex_20261108/deploy/graph_models`, 125 files)

| Item | `full_d64_l5_pre47_s10`, `full_d64_l5_pre48_s10` | `full_hex_pre46_s5` | Source |
|---|---|---|---|
| Script | `graph_train_full.py` (copy of the CV lane trainer, 1.7a) | `gnn_hex.py` | `lane_hex_20261108/source/gpu1_deploy_queue.sh:15-17` |
| Input | Cell graphs from pre-only detector peaks (`cell_graphs_pre47.npz`, `_pre48.npz`), 9 node features | `meta_ceiling_20261103/sweep/cell_graphs.npz` (pre46), 9 node features | launch lines; checkpoint fields `graphs`, `nfeat` |
| Network | d 64, 5 layers, context input zeroed | d 64, 5 layers, no context | checkpoint fields `d`, `layers`, `no_ctx` / `ctx` |
| Loss | Masked L1 on the log residual (log y - log base), three metrics | Same with loss weights 0,0,1 (HEX only) | `gnn_hex.py:1-4,83`; checkpoint `lw: '0,0,1'` |
| Optimizer, schedule | AdamW LR 1e-3, weight decay 0.01, OneCycleLR (max 1e-3, pct_start 0.1), clip 1.0 | Same | launch lines (`--lr 1e-3 --wd 0.01 --drop 0.1`); `gnn_hex.py:79-83` |
| Epochs, batch | 25 epochs, 64 frames per update | Same | launch lines; `gnn_hex.py:16` (`--bs` 64) |
| Dropout | 0.1 on the node update, 0.2 in the head | Same | `gnn_hex.py:32`; checkpoint `drop: 0.1` |
| Seeds | 10 each (0-9) | 5 (0-4) | launch lines (`--seeds`) |
| Snapshots | Epochs 21-25 of every seed are saved and averaged | Same | file names `_s{k}_e{21..25}.pt` |
| Training time | 803 s, 811 s (wall, GPU) | 401 s | HEX summary section 1 |

### 1.6 Attention-pool DINOv3-B (`lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46`)

All values from `launch.json` unless another source is named.

| Item | Value | Source |
|---|---|---|
| Backbone, initial weights | DINOv3-B/16 (HF) + the CD lane's full-data SSL5 (`lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt`) | `launch.json` (`family`, `ssl`) |
| Input | Whole frame, as 1.3 | family table in the runner |
| Pooling, head | 4 learned queries (init N(0, 0.02)), LayerNorm on the patch tokens, one `nn.MultiheadAttention` layer with 8 heads; the 4 outputs are concatenated (3,072) -> LayerNorm -> Linear(3072,128) -> GELU -> Linear(128,3) | `lane_backbone_20261108/source/bb_runner_attnpool_20261111.py:3-6,102-114,366-368` |
| What is trained | Full fine-tune; the pool and head train at the head LR (group "head", 13 tensors); `llrd` 0.0 | `launch.json` (`groups`, `llrd`) |
| Optimizer, LR, WD | AdamW; encoder 1e-5, head and pool 1e-3; weight decay 1e-4 | `launch.json` |
| Schedule, epochs | schedule10, 12 epochs | `launch.json` |
| Batch, accumulation | 16 per update, micro-batch 8 | `launch.json` (`micro`) |
| Drop path | 0.0 | `launch.json` |
| Loss | Recipe loss; flips only (`appear` 0, `shading` false, no window, no auxiliary targets) | `launch.json` |
| EMA / SWA | EMA 0.998, 0.999 from epoch 3 and SWA from epoch 6 are computed; tier p loads the raw track only | `launch.json`; manifest `hex_extra.attnp5` |
| Seeds, folds | Seed 46 only; folds 0-4 (`runs/attnpool_f{0..4}_dinov3b16_s46`, each from that fold's DINOv3-B SSL) and fold 99 | `launch.json` of f0 and f99; `jobs_D.tsv:21,23`, `jobs_G0.tsv:9`, `jobs_G1.tsv:7` |
| Used at inference | Raw epochs 8-11 and `final.pt` (epoch 12), single view, log mean; only the HEX column enters the HEX ridge | `members.py:163-174` |
| Training time | 2,602 s | `exit.json` |

### 1.7 CV lane models

#### 1.7a CV graph networks (`lane_cv_20261108/graph_models`, 200 files)

Network: Linear(9,64) node embedding; 5 message-passing layers (message MLP Linear-GELU-Linear, update LayerNorm-Linear-GELU, residual);
per-frame mean and std of the node states; head Linear(132,128)-GELU-Dropout(0.2)-Linear(128,3), zero-init; the 4 context inputs are
multiplied by 0 under `--no-ctx` (`lane_cv_20261108/source/graph_train_cv_20261108.py:44-51,63`).
Target: log y - log base prediction, masked L1 (`:20,82`).

| Item | Base ensemble `gfull_{noctx,pre47,pre48}` (80 files) | Plain TTA-peak quartet `full_tta_*` (60 files) | Order-loss quartet `gord_*` (60 files) | Source |
|---|---|---|---|---|
| Graphs | `cell_graphs.npz`, `_pre47.npz`, `_pre48.npz` | `lane_cv_20261108/graphs/tta_{s46,s47,s48,seeds3}_tta4.npz` (flip-TTA detector peaks) | Same four graph sets | checkpoint field `graphs`; `source/qC4.sh:9-10`, `qG.sh:6-7`, `qG2.sh:19` |
| d, layers | 64, 5 | 64, 5 | 64, 5 | checkpoint fields; launch lines |
| Epochs | 25 | 25 | 25 | launch lines |
| Optimizer | AdamW LR 1e-3, weight decay 0.01, OneCycleLR (pct_start 0.1), clip 1.0 | Same (weight decay 1e-2 and dropout 0.1 are script defaults, not on the launch line) | Same (script defaults 1e-2, 0.1) | `qC4.sh:7`; `graph_train_cv_20261108.py:14,78,82`; `qG2.sh:7`; `graph_order_cv_20261110.py:16` |
| Batch | 64 frames | 64 | 64 | `graph_train_cv_20261108.py:81-82` |
| Dropout | 0.1 (update), 0.2 (head) | Same | Same | `:45` |
| Loss | Masked L1 (`--order 0`) | Masked L1 | Masked L1 + 1.0 x pairwise BCE on the CV column, noise scale s = 0.065 | `qC4.sh:7`; `graph_order_cv_20261110.py:82-88,96`; `qG2.sh:7` |
| Seeds | noctx 0-9; pre47 0-2; pre48 0-2 | 0-2 per graph set | 0-2 per graph set | `qC4.sh:9-10`; `qG.sh:7`; `qC3.sh:6-11` |
| Snapshots | Epochs 21-25 saved and averaged | Same | Same | `graph_train_cv_20261108.py:83-85` |
| Precision | fp32 (no autocast in the script) | Same | Same | script |
| Training time | 5,955-6,817 s per seed on one CPU thread (noctx seeds 0-7) | not read | not read | `lane_cv_20261108/logs/gfull_noctx_s{0..7}.log:2` |

#### 1.7b Frozen token-statistic heads (`lane_cv_20261108/frozen_heads_full`, 60 files)

| Item | Value | Source |
|---|---|---|
| Encoder | Full-data recipe A s46 `epoch_20.pt`, frozen; features computed once from one unflipped view (bf16 forward, statistics in fp32) | `lane_cv_20261108/source/frozen_full_heads_20261109.py:1-3`; `token_stats_cv_20261108.py:1-5` |
| Arms (feature blocks) | `std` = std (768); `spectrum` = std + spectrum (1,536); `cov_sqrt` = std + cov_sqrt; `autocorr` = std + autocorr; PCA basis `lane_gpu2_20261106/frozen_s46/pca.npz` | `lane_gpu2_20261106/source/frozen_read_20261106.py:15`; checkpoint field `shapes` |
| Head | Per block LayerNorm, concatenation, Linear(D,128) -> GELU -> Linear(128,3); last layer N(0, 1e-3) | `frozen_read_20261106.py:48`; `frozen_full_heads_20261109.py:14` |
| Optimizer, LR, WD | AdamW (fused) LR 1e-3, weight decay 1e-4 | `frozen_full_heads_20261109.py:15` |
| Schedule | schedule10 form (`ratio(step, updates)`, `sched=10`) | `frozen_read_20261106.py:52-54` |
| Epochs, batch | 20 epochs, batch 16, clip 1.0 | `frozen_full_heads_20261109.py:16-20` |
| Loss | Recipe loss on the standardised log | `frozen_read_20261106.py:57-58` |
| Seeds | 0, 1, 2 per arm | `frozen_full_heads_20261109.py:13` |
| Rows | 8,952 | checkpoint field `rows` |
| Snapshots | Epochs 16-20 saved; the read averages their log predictions | `frozen_full_heads_20261109.py:21` |
| Training time | not found | - |

#### 1.7c Simulation-pretrained graph encoder (`meta_ceiling_20261103/sweep/sim_pred/sim2_l8_s{0,1,2}.pt`)

| Item | Value | Source |
|---|---|---|
| Training data | 24,000 simulated tessellation frames (`sim_graphs_v2.npz`); targets = exact log HEX, log area CV and log CD of the true cells, standardised; no real label is used | `sim_pred/sim2_l8.json`; `meta_ceiling_20261103/source/sim_graph_train2_20261105.py:1-2` |
| Network | Same graph network family, 9 node features, d 64, 8 layers, dropout 0.1 (update) and 0.2 (head) | `sim2_l8.json` (`args`); `sim_graph_train2_20261105.py:18` |
| Optimizer | AdamW LR 1e-3, weight decay 1e-2 (script constant), OneCycleLR pct_start 0.1, clip 1.0 | `:50,54` |
| Epochs, batch | 20 epochs, batch 64 | `sim2_l8.json`; `:50` |
| Loss | L1 | `:54` |
| Seeds | 3 (0-2) | `sim2_l8.json` |
| Use | Frozen; its pooled embedding of the real graphs feeds the CV/HEX ridge | CV summary section 2 |
| Training time | not found | - |

### 1.8 Cellpose-SAM `cpsam_models/cpsam_v2`

- The file itself was not trained by us. On the pod it is dated Jul 20, 1,233,586,851 bytes, mode 600 (`ls -la cpsam_models`).
- Use 1, as released: zero-shot segmentation. `seg_run.py` loads it with `CellposeModel(gpu=True, pretrained_model='cpsam_models/cpsam_v2')`
  and calls `model.eval(imgs, batch_size=8, ...)` with thresholds from the environment
  (`lane_hex_20261108/source/seg_run.py:15,18,54,62`). The KG method note gives the run settings as flow threshold 0.8, cell probability
  threshold -2, CLAHE, "no training" (`kg/wiki/methods/method-cellpose-sam-masks.md:5`; `kg/wiki/experiments/exp-hex-seg-shape-ridge.md:51`).
- Use 2, as initial weights: the five Phase-1 members start from its encoder and fine-tune it (1.1); the fine-tuned encoders are stored
  inside the member files, not in `cpsam_v2`. `docs/legacy_phase1/SSL_RELEASE_20260914.md:12` records "Initialization: Original cpsam_v2".
- No hash of the upstream release is on record in the files read, so identity with the public file was not verified here.

## 2. Hyperparameters that were varied, with the measured result

Sign convention: negative = lower error than the named baseline.

### 2.1 Whole-frame encoder recipe (recipe A and the family members)

| Hyperparameter | Values tried | Result | Baseline and folds | Source |
|---|---|---|---|---|
| Head LR and gradient clip | head 1e-3 / 3e-4 / 1e-4 x clip 1 / 3 | mean: 3e-4 clip 1 +0.052; 1e-4 clip 1 +0.072; 1e-4 clip 3 -0.008; 3e-4 clip 3 +0.077; 1e-3 clip 3 +0.114 | head 1e-3 clip 1 = 9.2403; fold 3, seed 42, mean pooling, SSL5, terminal epoch 20 (before std pooling and schedule10) | `phase2_lejepa5_sweep_20261013/audit.json` |
| Encoder LR lower / frozen | 3e-6; 0 (head only) | +0.038 (CD +0.312, CV -0.044, HEX -0.153); head only +0.866 | same baseline 9.2403, fold 3 | `phase2_lejepa5_tuning_20261012/audit.json`; arm manifests |
| Encoder LR x 5 | x 5 | train 8.30 -> 7.26, validation 9.11 -> 9.44 | recipe A, fold 0, seed 46 | `docs/ASSUMPTION_AUDIT_20261106.md:19` |
| Encoder LR with a CD-only loss | 1e-5, 3e-5 (CD loss only) | in place of plain s46 in the CD stack, strict: -0.0030 (1e-5), +0.0181 (3e-5). Alone: 6.3931 and 6.303 vs plain 6.3965 | incumbent CD stack 6.1588, folds 0+4, same run | ledger O-CDTRAIN-1E5-F04, O-CDTRAIN-3E5-F04 |
| LR schedule | schedule10 vs the previous cosine | -0.0399, -0.0985 | previous cosine schedule, fold 4 (fold 0 confirmation has no record) | `docs/ASSUMPTION_AUDIT_20261106.md:24` |
| Number of epochs (probe on stored runs) | per-epoch CD of recipe A e8-e20, ViT-L e8-e12 | recipe A single-epoch CD is lowest at e9-e12 and rises +0.004 to +0.008 per epoch to e20 (s46 e12 6.351, e20 6.434); ViT-L flat at e10-e12. The 1.5x-epochs arm was dropped without a run | five folds, per-fold scalar | ledger O-CDTRAIN-PROBE, D-CDTRAIN-002 |
| Pooling statistic | std, mean, max, min, mean_std, mean_std_min_max | std vs mean -0.0804 (CI -0.116, -0.048), four pairs; fold 4 seed 46: std 9.0479, mean_std_min_max 9.0557, mean_std 9.0613, max 9.0941, mean 9.1333, min 9.1343 | mean pooling; folds 0 and 4, seeds 46 and 47 | `recipe_A_20261027/decision.json`; `docs/ASSUMPTION_AUDIT_20261106.md:21,39` |
| Pooling: attention pool | 4-query attention pool vs std pool on DINOv3-B | member -0.005 (folds 0,1,2,4) / -0.0135 (folds 0,4); added to the CD stack +0.0047; HEX ridge column -0.0069 strict | plain DINOv3-B s46 raw epochs 8-12: 9.219 / 9.2485; HEX 38-column ridge 9.72662, five folds | KG `exp-bb-attention-pool-dinov3`; CD summary 4b; ledger O-HEX-001 |
| Head form | token MLP; HL-Gauss distribution head | token MLP -0.013 (CI -0.051, 0.027); distribution head DINOv3-B CD +0.008 | recipe A fold 0 seed 46; CD stack, five folds | `docs/ASSUMPTION_AUDIT_20261106.md:19`; CD summary 4b |
| SSL duration | 0, 1, 5, 10, 20 epochs | fold 3: 9.3017 / 9.2481 / 9.2403 / 9.2445 (0/1/5/10); fold 4: 9.1468 / 9.1333 / 9.1113 / 9.1200 (0/5/10/20); fold 0: 9.1811 / 9.2101 (5/10) | SSL5 on the same fold and seed; mean pooling; fold 3 seed 42, folds 4 and 0 seed 46 | `docs/ASSUMPTION_AUDIT_20261106.md:15` |
| SSL on/off for CPSAM 512 | no SSL vs SSL | standalone HEX epochs 8-12: 10.21 (no SSL) vs 10.24 (SSL) | fold 4 | KG `exp-hex-cpsam512-nossl` |
| Drop path | 0.2 on recipe A (with EMA/SWA tracks) | member +0.027; added at weight 0.125 -0.0021 | recipe A s46 fold 0, epochs 8-12: 9.0409 | KG `exp-bb-droppath-ema-swa` |
| Drop path + layer-wise LR decay | 0.3 + 0.9 on ViT-L | epoch-5 raw 10.40 vs about 9.3; stopped | ViT-L v1 at the same epoch, fold 0 | KG `exp-bb-vitl16-v2-regularised` |
| EMA / SWA | EMA 0.998 / 0.999 / 0.9995; SWA from epoch 6 | recipe A fold 0: swa6 9.071 vs raw 9.068. ViT-L two folds: swa6 9.115 vs raw 9.231 | raw track of the same run | backbone summary section 4 |
| Epoch window averaged at read time | epochs 8-20 vs 16-20 (recipe A); 8-20 vs 16-20 (DINOv3-B in the CD blend) | recipe A log mean of epochs 8-20: -0.045, three seeds -0.010 more. DINOv3-B: -0.0004 | recipe A epochs 16-20, seeds 46-48: 9.041, folds 0,4; CD blend 6.217, five folds | KG `exp-bb-checkpoint-average`, `exp-cd-family-snapshot-e8-20` |
| Recipe A snapshot (epochs 8-19) in CD | half e20 TTA + half e8-19 single view vs e20 TTA only | -0.004 as computed; zero under strict nesting (t1 6.1737 vs full 6.1743) | CD stack, five folds | CD summary 4a, section 6 |
| Best-epoch selection | pick the best epoch vs a fixed epoch | best-epoch pick flatters CD / CV / HEX by 0.019 / 0.027 / 0.017 | 13 recipe A runs, epoch fixed in advance | KG `exp-hex-checkpoint-selection-probe` |
| Loss weights per metric | CV x 3 (cvw3) on DINOv3-B | CV: -0.010 alone vs -0.009 recipe loss, weight 0 in the CV stack. As a CD member -0.048 vs -0.055 without it. HEX ridge: removal costs +0.0031 strict | best pipeline CV 10.151, five folds; HEX t1 9.71977 | KG `exp-cv-dinov3-cv-weighted-loss`; CD summary 4b; HEX summary 4a |
| Loss weights: HEX-weighted CPSAM 512 | HEX-weighted loss | +0.012 vs the recipe loss | CPSAM 512 with the recipe loss at share 0.2, fold 4 | KG `exp-hex-hexweighted-cpsam-loss` |
| Loss: CD x 3 (`cdw3`) | planned | never run | - | KG `exp-cd-cd-weighted-loss` |
| Loss: order loss on DINOv3-B | recipe loss + 1.0 x pairwise order loss on CV | CV stack 10.1143 with the 3-seed mean vs 10.1101 | six-component CV stack, five folds | CV summary 4b |
| Auxiliary targets | 15 label-free statistics at weight 0.2; green point auxiliary at 0.01 and 0.1 | +0.019; +0.027 and +0.047 | recipe A s46 fold 0 (9.0409); recipe A (folds not stated) | KG `exp-bb-aux-detector-targets`; `recipe_A_20261027/decision.json` |
| Target form | soft teacher targets; row re-weighting; area-SD target; label-distribution smoothing | +0.146 (fold 0); +0.024 (two folds); CV -0.0358 vs -0.0411; weight 0 (folds 0+4) | plain member / CV stack as named | backbone summary 4b; CV summary 4b |
| Augmentation: bundle (rotation, gamma, blur, noise) | on recipe A | -0.042 (CI -0.104, 0.019) | flips-only recipe A, fold 0, seed 46 | `docs/ASSUMPTION_AUDIT_20261106.md:18` |
| Augmentation: CV-lane bundle on DINOv3-B | on DINOv3-B | stack weight 0; inside the seed mean -0.030 vs -0.037 | CV stack, five folds | CV summary 4b |
| Augmentation: transpose + zoom x 1.12 (CD rescaled) | on recipe A | +0.116 | recipe A s46, fold 0, epochs 8-9 | backbone summary 4b |
| Augmentation: C-Mixup | on DINOv3-B | member -0.015; stack weight 0 (CD +0.0099, CV +0.0073, HEX +0.0153 when added) | plain DINOv3-B s46, folds 0,4 | backbone summary 4b; CD summary 4b |
| Augmentation: measured noise / optics | on recipe A | +0.004 (CI -0.043, +0.049) | matched reference 9.081, fold 0, seed 46 | KG `exp-gpu0-measured-augmentation` |
| Patch stride | DINOv3-B stride 8 vs 16 | member CD 6.88 vs 6.58 (as written); fold-0 member +0.471 vs +0.294 against the reference | recipe A reference, fold 0 | CD summary 4b; KG `exp-gpu2-dinov3-stride8` |
| Input view | 2x technician-window crop; legible-window crop; native tiles (CD only); canonical cell-size zoom; localised pooling | recipe A window CD +0.0026 (five folds); legible window CV +0.031 at w 0.25 (folds 0+4); tiles CD +0.255 (folds 0+4); canonical size +0.067, +0.235, zoom +0.088; localised pooling +0.071, +0.092 | plain whole-frame member or stack as named | CD, CV, backbone summaries 4b; `docs/ASSUMPTION_AUDIT_20261106.md:23-24` |
| Backbone size / family | ViT-L/16, ViT-H+/16, ConvNeXt-B/32 | ViT-L alone +0.215 raw vs recipe A (two folds), in the CD stack 6.1775 = -0.086 vs best pipeline; ViT-H+ -0.0034 on two folds (closed for cost); ConvNeXt weight 0.05, about 0.000 | recipe A member; best pipeline CD 6.2639 as computed, five folds | backbone summary 4a; CD summary 4 |
| Initial weights: synthetic pretraining | synthetic specular pretraining before fine-tuning | DINOv3-B member -0.072 (9.1606 vs 9.2326); ViT-L -0.068; recipe A -0.006 (CI -0.029, +0.016); stack reads CD -0.0031 (ViT-L) / -0.0017 (recipe A), CV +0.0035 to +0.0078 | plain member of the same family, five folds, s46 | backbone summary 4b |
| Seeds | recipe A s47 vs s46; third recipe A seed; ViT-L second seed; DINOv3-B seeds 49, 50 | +0.027 (CI +0.004, +0.047); CD -0.003 on two folds; ViT-L seed mean -0.0049 (five folds); CV four seeds -0.031 vs -0.037 | folds 0,4 epochs 8-12 (9.0166); CD / CV stacks as named | KG `exp-bb-seed-noise`; CD summary 4; CV summary 4b |
| Test-time flips | four-flip TTA vs single view | CD -0.008 over single view (DINOv3-B epochs 16-20); CV weight 0 | CD blend, five folds | KG `exp-cd-flip-tta-snapshot`; CV summary 4b |
| RoPE coordinate scale at inference | 1.414, 1.19 | member -0.022; on the best pipeline +0.002 / -0.001 | best pipeline, five folds, seeds 46, 47 | KG `exp-gpu0-rope-scale-inference` |
| Label fraction | 50 % of the labels | 9.165 vs 9.081 | recipe A, fold 0 | KG `exp-gpu0-label-fraction` |

### 2.2 Graph networks

| Hyperparameter | Values tried | Result | Baseline and folds | Source |
|---|---|---|---|---|
| Epochs | 40, 50, 80 vs 25 | 50 epochs: folds 0,4 -0.027, five folds +0.004 vs 25 epochs; 80 epochs +0.116 (s47) | 25-epoch TTA t0.25 trio (five folds -0.011 vs best pipeline CV 10.151) | KG `exp-cv-graph-longer-training` |
| Width, depth | d 96, 6 layers, 40 epochs vs d 64, 5 layers, 25 epochs | +0.012 | d64 l5 e25 on the same nodes, folds 0,4 | KG `exp-cv-larger-graph-net` |
| Number of seeds | 10 vs 3 | -0.001 | best pipeline graph component, five folds | KG `exp-hist-cell-graph-network` |
| Loss target | CV only; HEX only; order loss | CV-only -0.002 vs -0.017 multi-target; HEX-only mixed 0.25: -0.0026 (folds 0,4), +0.0003 (five folds), kept in the HEX chain; order-loss quartet as a second component 10.1101 vs 10.1143 without it | as named | KG `exp-hist-cv-only-graph-loss`, `exp-hex-hex-only-gnn`; CV summary 4a |
| Node peak threshold | 0.25 vs 0.35 | -0.011 alone; in the stack -0.037 vs -0.040 without it | best pipeline CV 10.151, five folds | KG `exp-cv-tta-t025-trio` |
| Node / edge inputs | window attention; image tokens at nodes (32, 8 dims); image context; whole-frame graph | +0.013 / +0.033; +0.165, +0.144 (32 dims), +0.041 (8 dims); HEX variants +0.0000 to +0.0035 on the chain; +0.005 | graphs of the time, as named | CV summary 4b; `docs/ASSUMPTION_AUDIT_20261106.md:28`; KG `exp-hex-gnn-variants` |

### 2.3 Detectors and frozen heads

| Hyperparameter | Values tried | Result | Baseline and folds | Source |
|---|---|---|---|---|
| Gaussian target sigma | 3.5 px, 5 px vs 8 px | Failed the green-track gate (sigma 3.5: jitter 0.162, miss 0.42; sigma 5: jitter 0.179). Both are among the 16 detectors of the HEX consensus; removing that consensus changes HEX by -0.0002 in the t3 ladder | 25 annotated frames; HEX tier ladder, five folds strict | `green_track_gpu3_notes.md:88,92`; HEX summary 4b |
| Pseudo-label self-training | noisy student 4,000 updates; consensus teacher | F1 0.859 (unchanged), misses +0.046; F1 +0.012; first pilot F1 0.642 -> 0.666 | 25 annotated frames / val-5 | `green_track_gpu3_notes.md:18,36,125` |
| Detector seeds | 16-detector consensus vs fewer | removal costs -0.0002 (free); detector seed mean for the graph +0.00003 | HEX t3 ladder / chain, five folds | HEX summary 4b |
| Loss on frozen token heads | CV-weighted, CV-only vs recipe loss | members 10.434 / 10.438 vs 10.424; stack weight 0 | frozen std head, five folds | KG `exp-cv-frozen-heads-cv-loss` |
| Frozen head statistic / form | quantile head; spectrum; autocorr; cov_sqrt; HL-Gauss | quantile -0.029 (CI -0.044, -0.013); spectrum +0.005; autocorr +0.014; cov_sqrt +0.019; HL-Gauss +0.036 (members); added to the pipeline -0.0085 to -0.0095 for the three adopted blocks | std head on the same frozen encoder 9.0417, folds 0,4 pooled | KG `exp-gpu2-frozen-pooling-heads` |

### 2.4 Phase-1 crop members

Records of Phase-1 hyperparameter screens were not searched beyond the deployed runs. Facts read:
- Per-fold best epochs of the 25-epoch seed recipe range from 16 to 25 (`final_candidate_20260904/plan.json`).
- CPSAM 512 central crop vs 256 crops, one epoch: 9.4455 vs 9.6909, fold 2 (`docs/ASSUMPTION_AUDIT_20261106.md:50`).

## 3. Hyperparameters with no experiment on record

Records checked for this section: `kg/wiki/_views/tried-and-closed.md` (all rows), a keyword search of `kg/wiki/{experiments,ideas,methods,
findings,decisions}` and of all 209 ledger lines for "weight decay", "warm-up", "batch size", "beta", "dropout", "LLRD", "drop path",
"resolution"; `docs/ASSUMPTION_AUDIT_20261106.md`; the five lane summaries; `docs/FINAL_SUMMARY_20261113.md`.

For recipe A, DINOv3-B/16 and ViT-L/16 (the whole-frame families), no experiment was found that varied:

- Weight decay (1e-4 in every run). Zero hits for the term in the KG and the ledger.
- Warm-up length (1 epoch). Zero hits.
- Effective batch size (16) or the accumulation layout. Zero hits.
- Optimizer type and AdamW betas / eps.
- Mixed-precision mode of training (bf16 in every run).
- The loss mixing constants (0.5 / 0.5 and the 0.2 scale) with all three metrics kept. The only loss changes on record are per-metric
  weights (2.1) and the CD-only form.
- The target transform (standardised log).
- Head width (128) and last-layer initialisation.
- The floor of the schedule (0.1) and the cosine length (10 epochs) once schedule10 was adopted.
- Head LR and gradient clip under the adopted recipe (std pooling, schedule10). The only sweep on record predates both (2.1, fold 3,
  seed 42, mean pooling).
- Encoder LR under the adopted recipe with the joint three-metric loss, other than x 5 on fold 0 (2.1).
- More than 20 epochs with the joint loss for recipe A; any epoch count other than 20 for DINOv3-B (plain) and other than 12 for ViT-L,
  cvw3 and the attention-pool member.
- Layer-wise LR decay on recipe A and DINOv3-B (tried only on ViT-L together with drop path 0.3).
- Drop path on DINOv3-B and, alone, on ViT-L.
- Whole-frame input resolution (native 972x1296 padded to 992x1312 in every run) and the CLAHE / percentile preprocessing constants.
- SSL settings other than duration and seed (lambda 0.02, LRs, batch 32, projector, view augmentation); the audit states this itself
  (`docs/ASSUMPTION_AUDIT_20261106.md:15`).

The backbone lane summary (section 4, "Never tried on recipe A") and `docs/FINAL_SUMMARY_20261113.md:132` list the same items for recipe A;
that list also names appearance augmentation, for which the runner has an `--appear` switch that was never run.

For the other families, within the records listed above:
- Detectors: no record of a change of LR, weight decay, batch (2) or update count (1,000) for the fine-tuned detectors.
- Frozen token heads: no record of a change of LR, weight decay, epochs or batch.
- Graph networks: no record of a change of LR (1e-3), batch (64) or the OneCycle shape; weight decay 0.1 with dropout 0.3 appears only in
  one node-token variant (`meta_ceiling_20261103/sweep/graph_sweep/g1_tk_crop8_reg.json`), whose result was not read.
- Simulation-pretrained graph encoder: only the configs `sim2_l8` and `sim2_l8b` exist in `sim_pred`; no comparison was read.

## 4. Not found or not verified

- AdamW betas and eps: not set in any script, not stored in any config.
- Phase-1 members: the configs of the five-fold OOF runs were not opened; the revision of `train_deploy_methods.py` that produced
  metric35 (sha 3c7135c3...) was not found.
- The exact multiplier target of the "encoder LR x 5" run (the audit gives only "x 5").
- A record of the schedule10 confirmation on fold 0 (the audit says there is none).
- Why the cvw3, attention-pool and ViT-L members run 12 epochs: no record read.
- Why the first `dinov3_b16_f99_s46` run ended rc 1, and why ViT-L s47 took 15,083 s against 6,844 s for s46: not looked up.
- Training time of the CV frozen token heads, the `full_tta_*` and `gord_*` graph quartets, the simulation nets and `sg2pref_s46`.
- Fold-run configs of cvw3 (`lane_cv_20261108/arms/dinov3_cvw3_f*_s46`, `lane_hex_20261108/runs/cvw3_f*_s47`) were not opened.
- Upstream hashes of `cpsam_v2` and of the DINOv3-B base weights: none on record in the files read.
- Results of the node-token graph variant with weight decay 0.1 / dropout 0.3.
- Phase-1 hyperparameter screens (legacy documents) were not searched.

## 5. Inconsistencies seen while reading (facts only)

- The backbone lane's queue comment for the attention-pool runs reads "worse on all three -> close" (`lane_backbone_20261108/jobs_D.tsv:20`),
  while the member is an adopted HEX ridge column (D-HEX-004).
- The Phase-1 standardisation constant for HEX is 0.2749 (`code/final_candidate_core.py:11`); recipe A's train-only log std for HEX is
  0.1602 (`fulldata_recipeA_20261028/regression_std_s46/manifest.json`).
- Phase-1 all-data fits use 8,998 rows of a 9,000-row cache; the Phase-2 full-data fits use 8,952 rows of 8,954.
- `final_candidate_core.py` `load_member` asserts `k_crops == 1` (`code/final_candidate_core.py:84`), and predbag2_35's stored config has
  `k_crops` 2. Which loader tier p uses for this member was not checked.
- The backbone summary's "never tried on recipe A" list includes the encoder and head learning rates only implicitly; two pre-recipe sweeps
  of head LR x clip and of encoder LR exist on the pod (2.1, first two rows).
