# Backbone lane (GPU 3), from 2026-11-08

Target: one model, two-fold (0, 4) member read <= 8.95 (recipe A 9.046), stretch 8.85, by 11-11 18:00 EST. Rules in `protocol.json`; one line per try in `RESULTS.tsv`. Times Eastern.

## What stored predictions already say

| Read, folds 0 and 4 | Value | Change |
|---|---:|---:|
| Recipe A member, epochs 16-20, seeds 46-48 | 9.041 | |
| Same, epochs 8-12 | 9.030 | -0.011 (12-epoch screens lose nothing) |
| Checkpoint average, epochs 8-20, one seed | 8.996 | -0.045 (EMA/SWA expectation) |
| Three seeds x epochs 8-20 | 8.986 | -0.055 |
| Pipeline's A (seeds 46+47, flip TTA, x0.99 scalar) | 8.978 | -0.063 |
| Hybrid B | 8.896 | |
| Best pipeline | 8.758 | |

Recipe A seed 47 against seed 46 on epochs 8-12 is +0.027 with a paired CI of +0.004 to +0.047, so paired CIs understate run-to-run noise.

## Directions (charter 11-08 evening: order by the metric lanes' gaps CV 0.12, CD 0.09, HEX 0.04; about 12 GPU-h per direction on folds 0 and 4)

| Direction | Why it can close a gap | GPU-h used (est.) | Status 11-08 21:45 |
|---|---|---:|---|
| ViT-L/16 (diverse member) | two folds added CD -0.020, CV -0.021 (raw); five folds exist (CD lane ran 1-3) | ~11 | v1 kept. v2 (drop path 0.3 + LLRD 0.9) stopped: epoch 5 raw 10.40 vs v1 ~9.3 |
| ViT-L window view (diverse member) | technician region at 2x pixels; crop members are hybrid B's gain | ~8 | two folds: raw 9.211 (+0.195), added -0.0085 (CV -0.018); fold 4 alone -0.014. On top of whole-frame ViT-L it adds CV only (-0.021 -> -0.028) and costs CD/HEX. Third version (recipe A on the window, folds 0 and 4) running since 09:49 EST |
| Auxiliary detector targets (recipe A + 15 label-free cell statistics) | puts detector geometry into the encoder; aims at CV and HEX | ~2 | continue: fold 0 running |
| Geometric augmentation (transpose + CD-equivariant zoom) | small recipe A change, CD only | 0 | behind the window member in slot B |
| Drop path on recipe A | none seen | ~5 | stopped: fold 0 neutral (+0.027, added -0.002) |
| CPSAM 512 | two folds added HEX -0.019, CD +0.023 | ~22 (inherited) | done on folds 0 and 4: HEX-only member; five folds are the HEX lane's call (SSL does not fit the 12 h rule) |

## Rebuild sheet for the 11-10 integration rehearsal (members a metric lane uses)

Run everything from `/workspace/clear-ec-research` with `/workspace/venv/bin/python`, one visible GPU. Paths below are relative to `experiments/`. `--fold -1` (added 11-09 10:30 EST) trains on all 8952 frames with three positive labels; its per-epoch "validation" is a fixed 256-row in-sample monitor, not a score.

### 1. DINOv3 ViT-L/16 whole frame (used by the CD lane; five folds = my folds 0 and 4 + the CD lane's folds 1-3; seeds 47/48 are the CD lane's)

| Item | Value |
|---|---|
| Weights | timm `vit_large_patch16_dinov3.lvd1689m` (PI-approved), `phase2_backbone_screen_20261001/hf_cache/hub/models--timm--vit_large_patch16_dinov3.lvd1689m/snapshots/30c1109559f65dea34316b0d4842d35c5771fe11/model.safetensors` |
| Fold checkpoints | `lane_backbone_20261108/runs/vitl16_avg_f{0,4}_s46/<track>/runs/fold{f}/dinov3_l16t/final.pt`, track in raw / swa6 / ema999 / ema998 (fold 4) or ema9995 (fold 0, not for use); epoch 12, fp32 state dict (tokens + head) with `log_mean`, `log_std`, `config`; 1.2 GB each |
| Fold SSL | `lane_backbone_20261108/runs/vitl16_avg_f{0,4}_s46/ssl/last.pt` |
| OOF predictions | same folders, `predictions_epoch_01..12.npz` (members read the log mean of epochs 8-12) and `predictions_final_tta.npz` (epoch 12, 4-flip TTA) |
| SSL command | `python experiments/lane_backbone_20261108/source/family_bb_20261108.py ssl --family dinov3_l16t --fold F --seed 46 --root R --workers 2` |
| Regression command | `python experiments/lane_backbone_20261108/source/bb_runner_20261108.py --family dinov3_l16t --fold F --seed 46 --root R --ssl R/ssl/last.pt --epochs 12 --micro 8 --ema 0.998,0.999 --swa-start 6 --tta-final --workers 2` |
| Full data | the two commands with `--fold -1` |
| Full-data time, one H200 alone | about 4.7 h: SSL about 2.1 h + regression about 2.6 h (CD lane GPU 0 measured 1.7 h + 2.1 h per fold on 7150 frames; x1.25 for 8952). Within the 12 h rule |
| Input / output | whole frame, CLAHE + 1-99 percentile, replicate-pad to 992 x 1312, ImageNet normalisation; std pool -> LN -> 128 -> GELU -> 3; prediction = exp(z * log_std + log_mean) |

### 2. CPSAM 512 (HEX-only member, used by the HEX lane on folds 0 and 4)

| Item | Value |
|---|---|
| Fold checkpoints | none: both folds were stopped after epoch 12 and the family runner keeps weights only for epochs 16-20. Fold SSL: `lane_gpu3_20261106/runs/cpsam_c512_f{0,4}_s46/ssl/last.pt` |
| OOF predictions | `lane_gpu3_20261106/runs/cpsam_c512_f{0,4}_s46/runs/fold{f}/cpsam_c512/predictions_epoch_01..12.npz` |
| Command | `python experiments/lane_gpu3_20261106/source/family_recipe_20261106.py all --family cpsam_c512 --fold F --seed 46 --root R` (20 epochs by default; read on epochs 8-12) |
| Full-data time | about 21 h alone (SSL 8.6 h and 43 min per epoch on 7150 frames, x1.25): **does not fit the 12 h rule**. No full-data mode in that runner. The deployable CPSAM is the HEX lane's no-SSL multi-target version |

### 3. Window members (not used by any lane yet)

- Window-view ViT-L: `lane_backbone_20261108/runs/vitl16_win2x_f{0,4}_s46/` (same layout as 1). Command = item 1's regression command plus `--window experiments/lane_backbone_20261108/data/window_topbox_20261108.npz --window-size 576,448 --window-scale 2 --window-jitter 32`, SSL from item 1. Full-data about 2.5 h alone (regression only). On folds 0 and 4 it adds CV -0.018 alone but only CV on top of item 1.
- Recipe A on the window: running (folds 0 and 4), read due about 20:00 EST 11-09.
- Window centres come from `meta_ceiling_20261103/sweep/legible_window.npz` (pre-only detector peaks); at test time the same detector pass must produce `top_box_x`, `top_box_y`.

## Daily continue / stop, 2026-11-09 09:00 EST

| Direction | GPU-h so far (shared GPU, approx.) | Read on folds 0 and 4 | Decision |
|---|---:|---|---|
| ViT-L/16 v1 (whole frame) | ~11 | member +0.098 (swa6); added -0.013 (CD -0.020, CV -0.021, raw) | kept for CD (on five folds the CV lane gives it weight 0); seeds s47/s48 run in the CD lane |
| ViT-L window view (2x technician region) | ~5 | fold 0 added -0.002 (CV -0.013); fold 4 running | continue to fold 4 only; stop if it adds nothing |
| ViT-L v2 (drop path 0.3 + LLRD 0.9) | ~2 | fold 0 epoch 5 raw 10.40 | stopped |
| Auxiliary detector targets on recipe A | ~5 | fold 0 member +0.019, added -0.002, residual r 0.99+ | **stop**: neutral; root-cause says detector-target routes have no headroom |
| Drop path on recipe A | ~5 | fold 0 +0.027, added -0.002 | stopped (11-08) |
| Geometric augmentation on recipe A | ~2 | fold 0 running | continue to fold 0 only; fold 4 held |
| CPSAM 512 (inherited) | ~22 | member +0.254; added -0.001 (HEX -0.019, CD +0.023) | done: HEX-only member, handed to the HEX lane |

## Arms

| Arm | Folds | Status |
|---|---|---|
| Drop path 0.2 + EMA + SWA from epoch 6 | 0 | Neutral: raw 9.068 (+0.027, three-seed +0.005), swa6 9.071; added -0.002. Not kept; fold 4 skipped. Inside a 12-epoch run SWA gains nothing over raw (prediction averaging of epochs 8-12: -0.022) |
| DINOv3 ViT-L/16 (timm, PI-approved), SSL5 + 12 epochs + averaging tracks | 0, 4 | **Kept as a member.** Two folds: swa6 9.115 (+0.098; fold 0 +0.150, fold 4 +0.046), raw 9.231 (+0.215). Added w 0.125: raw -0.013 (CD -0.020, CV -0.021), swa6 -0.010 (CD -0.014, CV -0.015). Train probe 7.5-7.9: overfits. CD lane runs folds 1-3 |
| ViT-L/16 v2: drop path 0.3 + layer-wise LR decay 0.9, from v1's SSL | 0, 4 | queued after ViT-L fold 4 (slot B) |
| Label-free auxiliary targets (15 detector-window statistics), weight 0.2 | 0 | Neutral: raw 9.060 (+0.019, three-seed -0.003), residual r 0.99+, added -0.002. Stopped; fold 4 skipped |
| ViT-L/16 v1 second seed (s47) | - | not run here: the CD lane trains s47 and s48 on five folds (coordinator, 13:45 EST) |
| Geometric augmentation: batch transpose p 0.5 + zoom up to x1.12 with CD x 1/s^2 | 0, 4 | queued (slot B) |

## CPSAM 512 (owned, from lane GPU 3)

| Read, epochs 8-12 | Fold 4 | Fold 0 |
|---|---:|---:|
| Member | 9.214 (recipe A 8.992): +0.222; CD +0.28, CV +0.32, HEX +0.07 | 9.327 (recipe A 9.041): +0.286; CD +0.26, CV +0.40, HEX +0.19 |
| Added to best pipeline, w 0.125 | -0.013 (CD +0.015, CV -0.020, HEX -0.033); w 0.25 +0.002 | two folds: -0.0006 (CD +0.023, CV -0.006, HEX -0.019) |

Two-fold verdict: a HEX-only member (HEX -0.019 at w 0.125; it hurts CD). Fold 0 stopped after epoch 12 at 04:49 EST 11-09.

Residual r with recipe A 0.89 / 0.93 / 0.95, train probe 7.49: it fits training frames harder and validates worse, like the other families. The curve is flat from epoch 7 (9.18-9.26), so fold 4 was stopped after epoch 12 (06:45 EST) to hand GPU 1 to the HEX lane; fold 0 stops after its epoch 12 too. Deployability: SSL 8.6 h + 12 epochs 8.6 h on one GPU exceeds the 12 h full-data rule as it stands.
