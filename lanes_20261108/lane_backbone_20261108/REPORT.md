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
| ViT-L window view (diverse member) | technician region at 2x pixels; crop members are hybrid B's gain | ~8 | two folds: raw 9.211 (+0.195), added -0.0085 (CV -0.018); fold 4 alone -0.014. On top of whole-frame ViT-L it adds CV only (-0.021 -> -0.028) and costs CD/HEX. Third version: recipe A on the window, **kept** (see next row) |
| Recipe A (CPDINO s8) on the 2x window | the strongest family on the technician region at 2x pixels | ~9 | **Kept.** Folds 0+4 raw 9.101 (+0.084; HEX -0.015 alone), residual r 0.93/0.97/0.98; added -0.0099 (CD -0.010, CV -0.012, HEX -0.008); on top of whole-frame ViT-L it adds -0.006 (stack -0.0189). Fold 1 done on GPU 2 (CV lane, 4.1 h); fold 2 on GPU 3 here since 06:02 EST (needed the zero-label runner fix); fold 3 on GPU 0 (CD lane, since 01:49 EST); same command, roots here. Reads: CD lane stack -0.0137 on folds 0+4, -0.0041 on four folds, **+0.0026 on five folds (not adopted for CD)**; HEX lane chain +0.0002 (not adopted); CV lane no help. **Closed**; full-data build cancelled |
| Auxiliary detector targets (recipe A + 15 label-free cell statistics) | puts detector geometry into the encoder; aims at CV and HEX | ~2 | continue: fold 0 running |
| Geometric augmentation (transpose + CD-equivariant zoom) | small recipe A change, CD only | 0 | behind the window member in slot B |
| Drop path on recipe A | none seen | ~5 | stopped: fold 0 neutral (+0.027, added -0.002) |
| CPSAM 512 | two folds added HEX -0.019, CD +0.023 | ~22 (inherited) | done on folds 0 and 4: HEX-only member; five folds are the HEX lane's call (SSL does not fit the 12 h rule) |

## Rebuild sheet for the 11-10 integration rehearsal (members a metric lane uses)

Run everything from `/workspace/clear-ec-research` with `/workspace/venv/bin/python`, one visible GPU. Paths below are relative to `experiments/`. `--fold -1` (added 11-09 10:30 EST) trains on all 8952 frames with three positive labels (SSL and regression); its per-epoch "validation" is a fixed 256-row in-sample monitor, not a score. Agreed with the CD lane 11-10: this is the same 8952 rows as its `--fold 99` (`experiments/fulldata_recipeA_20261028/fulldata_regression_split.npz`, the two HEX-zero rows held out), which the CD-stack members use with the CD lane's scripts.

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
- Recipe A on the window (kept on folds 0+4; CD lane stack -0.0041 on four folds): five folds in `lane_backbone_20261108/runs/recipeA_win2x_f{0..4}_s46/` (fold 2 finishing 11-10 ~16:00 EST). Fold command = item 1's regression command without `--family`/`--ssl` (recipe A SSL5 is automatic per fold) plus the window flags above. Full-data build queued on GPU 3: `--fold -1 --ssl experiments/fulldata_recipeA_20261028/ssl5_s1003/run/last.pt` (the all-data SSL of recipe A's full-data member, all 8954 images, label-free), `--keep-epochs 8,9,10,11,12`, root `runs/recipeA_win2x_full_s46`; about 6 h alone.
- Window centres come from `meta_ceiling_20261103/sweep/legible_window.npz` (pre-only detector peaks); at test time the same detector pass must produce `top_box_x`, `top_box_y`.

## GPU 3 to the freeze (coordinator, 11-10 15:00 EST)

1. CD lane full-data ViT-L s47 (its script, fold 99, its root), waiting for s46's full-data SSL (~17:00 EST), ~2.6 h.
2. Room for the cell-route lane's end-to-end inference run (small).
3. CV lane order-loss DINOv2 B/14 (qD2O.sh, five folds + full data, ~10-11 h): running since 15:19 EST (slot B).
4. CV lane order-loss DINOv3 seed 49 folds 1-3 (qS49g3.sh): waits for the CV lane's go (~17:15 EST).
5. **Shift harness (PI, new direction):** leave-one-source-out (P1 = 'NNNN-YY' old, P2 = 'NNNN-NNNN aaaa' new), DINOv3-B/16 SSL5 + 12 epochs on one source, read on its hold-out and on the other source; shift penalty per metric. Slot D: SSL for both runs done; both regressions died at start (runner read .blocks, HF DINOv3 uses .layer; fixed 17:25 EST); regression only from the kept SSL, A since 17:27 EST; B held behind source/gpu3_launch_20261110.sh until the deploy build ViT-L s47 (pid 395437, ETA ~23:15 EST) ends (coordinator: it has priority over every experiment on GPU 3). Split files in data/ for other lanes. **Step 1 result (11-11 22:40 EST): no HEX-specific penalty.** Penalties (raw, epochs 8-12): P1 rows CD +3.37 / CV +5.95 / HEX +3.49, P2 rows CD +3.67 / CV +6.47 / HEX +0.78; biases mirror (each model predicts toward its own source's label level); removing the mean log offset (oracle diagnostic) leaves CD +0.50/+1.46, CV +0.42/+0.73, HEX +0.33/+0.35, so the shift is mostly a level offset between sources. Step 2 (only if HEX shows a clear penalty): appearance augmentation first, then label-free member-bias checks; test-time subspace alignment last, spare time only, measurement only (PI: not a freeze candidate unless the PI decides). Step 2 prep done (not run): `--appear P` in bb_runner (raw-frame gamma, contrast, blur or sharpening, noise before CLAHE; P=0 pixel-identical to recipe A). Raw appearance differs by source: P2 is brighter (median 120 vs 83), flatter (contrast 39 vs 52), softer (focus 118 vs 204) and has no black border (P1 median 10 % black).
6. **Shading correction, then appearance augmentation (coordinator 11-11 03:00Z, PI: no idle GPU):** `--shading` (raw frame divided by a sigma-64 px blur, floor 8, before CLAHE) and `--appear 0.5`, each read on the shift harness run A direction (cross-source bias and MAPE against shift run A) and on folds 0+4 (member against DINOv3-B s46 epochs 8-12, and in the lanes' stacks). Slot D after shift B: shade_srcA, shade_f0, shade_f4, appear05_srcA, appear05_f0, appear05_f4 (~8.5 GPU-h, box 12). Stop rule: weight 0 and no bias change. **Closed 11-11 23:24 EST by the coordinator's rule 1:** shade_srcA raised the cross-source offset (mean |log| 0.0825 -> 0.0959, cross MAPE worse on all three metrics, in-source unchanged), so shading does not explain the source offset; appearance augmentation closed with it, never run.
7. **Ensemble distillation (coordinator 11-11, KG idea-ensemble-distillation): closed 03:05 EST.** One DINOv3-B trained to exp(0.5 log y + 0.5 log candidate-OOF): folds 0+4 two-fold 9.001 (raw) against plain DINOv3-B 9.2485 and the integrated candidate 8.6876, so it closes 44 % of the gap (EMA/SWA 43-47 %), below the half-gap rule; CV closes most (64 %), HEX least (30 %). The read is slightly optimistic (teacher OOF saw the held-out labels). Folds 1-3 stopped (f1 killed by pid after 9 min on GPU 2).
8. **C-Mixup (coordinator 11-11, KG idea-mixup-cutmix):** runner copy bb_runner_cmix_20261111.py; cmix_f4 on GPU 0, cmix_f0 on GPU 1; read against plain DINOv3-B s46 epochs 8-12; worse on all three -> close without the stack read. **Closed 11-11 03:23 EST:** member read -0.015 (CD +0.026, CV -0.027, HEX -0.045) but weight 0 in all three stacks (CD +0.004..+0.011, CV +0.0073, HEX +0.0153); folds 1-3 killed by pid.
9. **End-to-end attention pooling (coordinator 11-11, KG idea-attention-pooling-whole-frame):** runner copy bb_runner_attnpool_20261111.py (4 learned queries, one 8-head cross-attention, concatenated into recipe A's head); attnpool_f0 GPU 3, attnpool_f4 GPU 0; read vs plain DINOv3-B s46 epochs 8-12, then to the CV and HEX lanes. **Closed 11-11 05:05 EST:** member read slightly better (folds 0+4 -0.013, four folds -0.005; CD -0.04, CV +0.02-0.03) but weight 0 in all three stacks (CD five folds +0.003..+0.005 against the s46/s47 DINOv3-B seed mean, CV 0.000, HEX +0.0055).



## Freeze-day summary, 2026-11-11 (written 17:20 EST)

| Direction (decision id) | GPU-h (approx.) | Read | Status |
|---|---:|---|---|
| Attention-pool full-data build (D-HEX-004 column) | 0.8 | verified: strict load, held-row recompute 0.0, sanity r 0.985-0.987 | delivered (ledger A-BB-ATTNP-F99), in the HEX freeze candidate |
| Synthetic renderer v1 -> v3 (coordinator review) | 0 (CPU) | v1 inverted polarity (caught by the coordinator), v2 3-6x too contrasty; v3 matched by measured statistics (edge sharpness, cell contrast, cell-to-cell steps) and HEX label quantiles | done; 50k frames data/synth/synth_v3 |
| Synthetic zero-shot columns | 0.2 | label r CD 0.61 / CV 0.42 / HEX 0.43 (beats CPSAM-Konan and detector on CD, CV) | closed in all three stacks (O-BB-SYN-ZS) |
| synthft3 DINOv3-B, five folds (D-SYN-001) | ~7 | member 9.1606 vs plain 9.2326 (-0.072), every fold better | closed: CV swap +0.0035 strict, CD/HEX weight 0; full-data build verified as a spare (A-BB-SYNFT3-F99) |
| Synthetic-pretrained ViT-L s46, folds 0+4 (D-SYN-002) | ~6 | running; fold 0 lands ~17:50 EST, fold 4 ~18:25 EST | CV lane read armed (A-CV-SYN2-READ) |
| Synthetic-pretrained recipe A s46, folds 0+4 (D-SYN-003) | ~20 (running) | pretraining ~18:00 EST, fine-tunes ~03:00-03:30 EST 11-12 | CD lane builds the swapped A (A-CD-SWAPA-PLAN) |

Lesson for this lane (L1 again, now with the strongest single-member gain of the week): a member 0.07 better alone on five folds still got no stack gain, because the stacks already hold members with the same errors. D-SYN-002/003 test the only remaining form, lifting the members that carry the stacks.

## Daily continue / stop, 2026-11-11 09:00 EST (written 05:15 EST; no lane GPU after 05:08 EST)

| Direction | GPU-h (approx.) | Read | Decision |
|---|---:|---|---|
| Window view (recipe A win2x, five folds; ViT-L win2x folds 0+4) | ~20 | CD stack -0.014 on folds 0+4; HEX chain +0.0002 | done 11-10: CD lane uses it; no further runs |
| Target denoising v1 (soft teacher) / v2 (row weights) | ~6 | v1 +0.146 (CD +0.45); v2 +0.024, added -0.0025 | stopped 11-10 |
| Shift harness, step 1 (leave-one-source-out) | ~5 | no HEX-specific penalty; penalties CD +3.4/+3.7, CV +6.0/+6.5, HEX +3.5/+0.8; 85-90 % is a source level offset; candidate OOF offset by source ~0 | done; split files in data/ for other lanes |
| Shading correction (shift harness run A) | ~0.9 | cross-source offset 0.0825 -> 0.0959 (worse) | stopped (coordinator rule 1) |
| Appearance augmentation (--appear) | 0 | not run | stopped with shading (rule 1) |
| Test-time subspace alignment | 0 | not run | stopped (coordinator: a level offset cannot be fixed by feature alignment without labels) |
| Ensemble distillation (one DINOv3-B to the candidate) | ~1.9 | folds 0+4 9.001 vs 9.2485 / 8.6876: 44 % of the gap | stopped (half-gap rule); no larger student |
| C-Mixup | ~2.1 | member -0.015; weight 0 in CD, CV, HEX stacks | stopped |
| End-to-end attention pooling (4 queries) | ~4.3 | five-fold member 9.2329 vs 9.2326; weight 0 in CD, CV, HEX | stopped |
| Builds for other lanes | ~7 | full-data ViT-L s47 (CD), full-data order DINOv3 s47/s49 (CV): all completed | done |

Nothing of this lane is running. GPU 3 went to the HEX lane at 05:08 EST (coordinator); GPUs 0-2 went back to the CD and HEX lanes at 04:17 EST.

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
| Geometric augmentation: batch transpose p 0.5 + zoom up to x1.12 with CD x 1/s^2 | 0 | Closed: stopped at epoch 10 for ViT-H+; epochs 8-9 raw 9.157 (+0.116), swa6 9.081 |
| CD lane's DINOv3 ViT-H+/16 fold 3 (their recipe, their folder) | 3 | stopped 01:52 EST 11-10 in SSL: coordinator closed ViT-H+ on folds 0+4; partial run archived in the CD lane folder |
| Target denoising v1: soft targets, 0.5 label + 0.5 nested teacher in log space | 0 | **Stopped** at fold 0 epoch 9: epochs 8-9 9.186 (+0.146, CD +0.45: the teacher is weak on CD); fold 4 stopped too. Was: | fold 0 running on GPU 3 since 23:11 EST 11-09 (micro 2); fold 4 on GPU 0 via the CD lane from ~07:20 EST 11-10, both versions side by side, ETA ~14:30-15:30 EST; early read at epochs 8-9 |
| Target denoising v2: rows down-weighted by label-teacher disagreement (top 10 %, floor 0.2) | 0, 4 | **Closed (level):** two folds 9.041 (+0.024; three-seed +0.010), residual r 0.98-0.999, added -0.0025 (HEX -0.010). Was: | fold 0 running on GPU 3 since 23:11 EST 11-09 (micro 2); fold 4 on GPU 0 via the CD lane from ~07:20 EST 11-10, both versions side by side, ETA ~14:30-15:30 EST; early read at epochs 8-9 |

The target-denoising teacher is leak-free by construction: for outer fold f and a training row in fold g it is a gradient-boosted model fit on the three folds outside {f, g}, on 59 strictly label-free columns (image statistics, zero-shot Cellpose-SAM mask statistics, segmentation-only cell statistics; built by the cell-route lane with my builder logic). My first 46-column teacher was replaced before any run: its detector features come from detectors initialised from recipe A fold encoders. No image-model OOF can meet the rule, so the teacher is a cell reader, not the stack: plain OOF CD 7.45, CV 10.72, HEX 10.22.

## CPSAM 512 (owned, from lane GPU 3)

| Read, epochs 8-12 | Fold 4 | Fold 0 |
|---|---:|---:|
| Member | 9.214 (recipe A 8.992): +0.222; CD +0.28, CV +0.32, HEX +0.07 | 9.327 (recipe A 9.041): +0.286; CD +0.26, CV +0.40, HEX +0.19 |
| Added to best pipeline, w 0.125 | -0.013 (CD +0.015, CV -0.020, HEX -0.033); w 0.25 +0.002 | two folds: -0.0006 (CD +0.023, CV -0.006, HEX -0.019) |

Two-fold verdict: a HEX-only member (HEX -0.019 at w 0.125; it hurts CD). Fold 0 stopped after epoch 12 at 04:49 EST 11-09.

Residual r with recipe A 0.89 / 0.93 / 0.95, train probe 7.49: it fits training frames harder and validates worse, like the other families. The curve is flat from epoch 7 (9.18-9.26), so fold 4 was stopped after epoch 12 (06:45 EST) to hand GPU 1 to the HEX lane; fold 0 stops after its epoch 12 too. Deployability: SSL 8.6 h + 12 epochs 8.6 h on one GPU exceeds the 12 h full-data rule as it stands.
