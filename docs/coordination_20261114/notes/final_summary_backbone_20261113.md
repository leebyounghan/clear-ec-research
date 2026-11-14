# Backbone lane: final summary (2026-11-13, written 21:55 EST)

Lane folder: `/workspace/clear-ec-research/experiments/lane_backbone_20261108/` (below: `LB/`). Paths without a prefix are relative to `/workspace/clear-ec-research/experiments/`.
Running: nothing. No GPU job, keepalive, waiter, loop or cron of this lane is active. My last keepalives (GPUs 1-3) stepped aside by themselves at 20:12-20:13 EST when D-PKG-007 started [O-BB-KEEPALIVE-YIELD], and I did not restart them [D-GPU-002].
Records: `LB/RESULTS.tsv` (105 rows), `LB/REPORT.md`, `LB/reads/*.json`, and `lane_backbone_notes.md` in this notes folder.

## 1. Outcome

Frozen candidate: 8.66336 as computed / 8.66867 strict, against the best pipeline's 8.73627 (scalars refit) [O-VER-002]. With the cell-route step it is 8.66789 strict [O-VER-004]. Tier p's strict mean is 8.6682 [D-PKG-003].

- **Attention-pool DINOv3-B column in the HEX 39-column joint ridge [D-HEX-004].** HEX strict 9.71977 vs 9.72662 for the 38-column ridge (−0.0069) and vs 9.80262 for the best pipeline [O-HEX-001]. Removing it from tier t2 costs +0.0080 [O-HEX-010]. Its full-data build (`attnpool_f99`) was verified: strict load, held-row recompute max rel diff 0.0 [A-BB-ATTNP-F99]. Tiers p, t1, t2 and full load it; t3 does not.
- **DINOv3 ViT-L/16 whole-frame member.** My folds 0 and 4 (s46) are part of the ViT-L slot's five-fold OOF in D-CD-001. CD strict is 6.1743 vs 6.2610 for the best pipeline [D-CD-001]. Removing the ViT-L slot (tier t1 → t2) costs +0.0138 CD strict [D-PKG-003]. The deployed ViT-L files are the CD lane's full-data s46/s47 builds; s47 was trained on GPU 3 through my queue.
- **Builds run on my card for other lanes:** full-data order-loss DINOv3 s47/s49 for the CV lane. The deployed CV stack gives them weight 0 [O-DEPLOY-001].
- **Everything else this lane tried is closed (section 4).** That includes the three synthetic-pretraining routes [D-SYN-001, D-SYN-006, D-SYN-008].

## 2. Models

Sizes come from `ls`. A "track" is raw / ema998 / ema999 / swa6, each with its own `final.pt`. **P** marks files the tier-p package loads. The tier-t3 package loads nothing from this lane.

| Path | Role | What | Size |
|---|---|---|---|
| **P** `LB/runs/attnpool_f99_dinov3b16_s46/raw/runs/fold99/dinov3_b16/{epoch_08..11,final}.pt` | deployed in p, t1, t2, full (not t3) | full-data attention-pool DINOv3-B s46, HEX column attnp5 | 5 × 353.8 MB fp32. The package holds fp16 copies of 176.9 MB, hard-linked (5 links) |
| `LB/runs/attnpool_f99_dinov3b16_s46/{ema998,ema999,swa6}` | unused | other tracks of the same build | 353.8 MB each |
| `LB/runs/attnpool_f{0..4}_dinov3b16_s46` | candidate input | fold members whose OOF is the attnp5 column of D-HEX-004 (`lane_hex/source/joint_fit.py:37`, `joint_deploy.py:110`) | 353.8 MB per track |
| `LB/runs/vitl16_avg_f{0,4}_s46/<track>/runs/fold{f}/dinov3_l16t/final.pt` | candidate input | ViT-L s46 folds 0/4, ViT-L slot of D-CD-001. The raw f0 final.pt is also in the fold-0 test manifest | 1.21 GB per track. The f0 `ema9995` track is "not for use" |
| `LB/runs/vitl16_avg_f{0,4}_s46/ssl/last.pt` | SSL | ViT-L SSL5 for folds 0/4; init for synthetic ViT-L folds 0/4 | 1.21 GB each |
| **P** `lane_cd_20261108/arms/vitl16_f99_s47` (CD lane folder) | deployed in p, t1 | full-data ViT-L s47, trained on GPU 3 by my queue with the CD lane's script and root (first try rc 1 at 22:01 UTC 11-10, relaunched) | CD lane's records |
| CV lane folder, full-data order DINOv3 s47/s49 | built here, not deployed | CV lane's script and roots | CV lane's records |
| `LB/runs/synthft3_f99_dinov3b16_s46` | closed, reusable | verified full-data synthetic-pretrained DINOv3-B [A-BB-SYNFT3-F99] | 343.1 MB per ckpt; raw has 5 |
| `LB/runs/synthpre3_f{0..4,99}_dinov3b16_s46/pretrain.pt`, `synthft3_f{0..4}_dinov3b16_s46` | closed [D-SYN-001] | DINOv3-B synthetic pretraining and fold fine-tunes | 343.1 MB each / per track |
| `LB/runs/synthpre_vitl_f{0..4}_s46/pretrain.pt`, `synthft_vitl_f{0..4}_s46` | closed [D-SYN-006] | ViT-L synthetic pretraining and fold fine-tunes | 1.21 GB each / per track |
| `LB/archive/synthvitl_f99_dsyn006_20261112/` | closed, partial | full-data synthetic ViT-L: finished pretrain.pt, fine-tune stopped after epoch 9 (epoch_08.pt, no final) | 1.21 GB each |
| `LB/runs/synthpre_ra_f{0..4}_s46/pretrain.pt`, `synthft_ra_f{0..4}_s46`, `synthft_ra_f0_s47` | closed [D-SYN-008] | recipe A (cpdino_b8) synthetic pretraining and fine-tunes | 343.3 MB / final.pt 343.2 MB (raw only) |
| `LB/runs/recipeA_win2x_f{0..4}_s46` | closed | recipe A on the 2x technician window | 343.2 MB per track |
| `LB/runs/vitl16_win2x_f{0,4}_s46`, `vitl16_dp03_llrd09_f0_s46` | closed | ViT-L window view; ViT-L with drop path 0.3 + LLRD 0.9 | 1.21 GB per track |
| `LB/runs/{cmix,distill}_f*_dinov3b16_s46`, `tile_cd_f{0,4}_s46`, `droppath02_avg_f0_s46`, `aux_legible15_w02_f0_s46`, `geo_t05_z112_f0_s46(b)`, `target_{soft05,rowweight}_f{0,4}_s46`, `shift_src{A,B}_dinov3b16_s46`, `shade_*` | closed | closed arms (section 4) | ~343 MB (B/16, cpdino) per track |
| `lane_gpu3_20261106/runs/cpsam_c512_f{0,4}_s46/ssl/last.pt` | closed (HEX-only member, handed over) | CPSAM 512 fold SSL. No fold checkpoints were kept (stopped after epoch 12) | not recorded |

## 3. Data

| Path | What | Regenerable? | Other readers |
|---|---|---|---|
| `datasets/clear_ec_official/` (incl. `cache_native_meta_512.npz`) | official frames and labels | no | everyone |
| `ssl_screen_20260911/preflight/fold{0,1}.npz`, `overnight_target_20260912/preflight/fold{2,3}.npz`, `target_fold4_20260912/preflight/fold4.npz` | the five-fold split, read by `bb_runner`/`family_bb` | no (it defines every OOF) | all lanes |
| `fulldata_recipeA_20261028/fulldata_regression_split.npz` | fold 99 (8,952 training rows, the 2 HEX-zero rows held) | from the split rule | every full-data build |
| `oof_recipeA_20261027/fold2_zero_policy_addendum.json` | zero-label policy (fold 2: two HEX = 0 rows) | doc | bb_runner fold-2 fix |
| `LB/data/synth/synth_v3_frames.u8` (62,985,600,000 B = 50,000 × 972 × 1296 u8) + `synth_v3_labels.npz` (5.9 MB) | synthetic specular set v3 with exact CD, area CV and HEX; the last 1,000 frames are a hold-out | yes, `LB/source/synth_specular_20261111.py`: per-frame `SeedSequence([SEED0, i])`, never regenerated to confirm | synthetic pretraining only |
| `LB/data/synth/_archive_v1_inverted_polarity/`, `_archive_v2_too_contrasty/` | rejected renders | yes, not wanted | none |
| `LB/data/synth_zeroshot_f0pre_20261111.npz` | zero-shot predictions of the fold-0 synthetic model on all real frames | `synth_zeroshot_20261111.py` | `lane_hex/source/strict_joint_extra.py:10` |
| `LB/data/window_topbox_20261108.npz` | 2x window boxes | `make_window_20261108.py` from `meta_ceiling_20261103/sweep/legible_window.npz` | win2x runs |
| `LB/data/source_split_20261110.npz`, `split_src{A,B}_20261110.npz` | leave-one-source-out splits (P1 'NNNN-YY', P2 'NNNN-NNNN aaaa') | `make_source_split_20261110.py` | offered to lanes; readers unknown |
| `LB/data/aux_legible15_20261108.npz`, `teacher_*.npz`, `rowweight_*.npz` | auxiliary targets and target-denoising teachers/weights (LF = label-free cell-route features) | `make_aux`, `make_teacher*`, `make_nested_teacher`, `rowweight_from_teacher` | none |
| `LB/runs/<arm>/<track>/runs/fold{f}/<family>/predictions_epoch_XX.npz`, `predictions_final_tta.npz` | per-epoch OOF of every arm | retraining only | CD: `cd_tiers_20261112.py:27`, `cd_read_20261108.py:45,58`. CV: `cv_stack5_20261108.py:88,110,221`. HEX: `chain.py:252-254`, `joint_fit.py`, `joint_deploy.py` |
| `LB/reads/*.json` | read outputs | `LB/source/read_*.py` | ledger lines cite them |
| `integration_rehearsal_20261110/out/integrated_oof.npz` (P_main) | 11-10 rehearsal candidate, the comparator in my distillation, synthft3 and decile reads | not mine | see section 6 |

## 4. Ideas

Numbers are member reads (MAPE, epochs 8-12, raw) against the named plain model on the same rows, unless marked strict stack. (a) adopted, (b) closed, (c) open or partly tested.

(a)
- **idea-dinov3-vitl16-member:** in the CD stack [D-CD-001]; removing the slot costs +0.0138 CD strict [D-PKG-003]. Alone it is worse than recipe A (two folds +0.215 raw, +0.098 swa6). CV gives it weight 0.
- **idea-attention-pooling-whole-frame (as a HEX ridge column):** −0.0069 HEX strict (39 vs 38 columns) [O-HEX-001, D-HEX-004]. As a stack member it got weight 0 in CD, CV and the HEX chain (CD five folds +0.003..+0.005) [RESULTS row 70].
- **idea-checkpoint-log-mean-averaging:** recipe A folds 0+4, log mean of epochs 8-20 vs epochs 16-20: −0.045; three seeds −0.010 more [RESULTS rows 2-3]. The CD lane's snapshot form uses the same idea, and tier t1 drops that snapshot at no strict cost (6.1737 vs 6.1743).

(b)
- **idea-vitl-window-view-2x:** two-fold member +0.195; on top of the whole-frame ViT-L it adds CV only.
- **idea-recipeA-window-crop-2x:** CD five-fold stack read +0.0026 (6.1751 vs 6.1726, CD lane 11-10), HEX chain +0.0002 [RESULTS rows 36-37].
- **idea-drop-path-recipeA (0.2, with EMA/SWA):** fold 0 +0.027, added −0.002.
- **idea-layerwise-lr-decay-vitl (drop path 0.3 + LLRD 0.9):** epoch-5 raw 10.40 vs ~9.3 for v1; stopped.
- **idea-auxiliary-detector-targets (15 label-free statistics, weight 0.2):** fold 0 +0.019; residual r with recipe A 0.99+.
- **idea-geometric-augmentation-cd-equivariant (transpose + zoom ×1.12, CD × 1/s²):** fold 0, epochs 8-9, +0.116.
- **idea-mixup-cutmix (C-Mixup):** member −0.015; weight 0 in all three stacks (CD +0.004..+0.011, CV +0.0073, HEX +0.0153).
- **idea-cpsam512-member:** two-fold member +0.254; added at weight 0.125: HEX −0.019, CD +0.023. HEX-only; five folds were the HEX lane's call.
- **idea-target-denoising-soft-teacher:** fold 0 +0.146 (CD +0.45).
- **idea-target-row-reweighting:** two folds +0.024, added −0.0025.
- **idea-ensemble-distillation:** folds 0+4 9.001 vs plain DINOv3-B 9.2485, i.e. 44 % of the gap to the 11-10 rehearsal candidate (8.6876); below the half-gap rule.
- **idea-leave-one-source-out-shift:** no HEX-specific penalty [F-SHIFT-001]. Penalties CD +3.4/+3.7, CV +6.0/+6.5, HEX +3.5/+0.8 MAPE points (P1/P2); 85-90 % of it is a per-source level offset.
- **idea-shading-correction:** cross-source offset 0.0825 → 0.0959 (worse) [RESULTS row 50].
- **idea-native-tile-cd-only:** CD +0.255 vs plain DINOv3-B 6.689 (folds 0+4) [A-BB-TILE-F04].
- **idea-synthetic-zero-shot-columns:** closed in all three lanes (HEX +0.0011, CV +0.0152) [O-BB-SYN-ZS]. Zero-shot label r CD 0.61 / CV 0.42 / HEX 0.43.
- **idea-synthetic-pretraining-dinov3b:** five folds 9.1606 vs 9.2326 (−0.072, every fold lower); CV swap +0.0035 strict, CD and HEX weight 0 [D-SYN-001].
- **idea-synthetic-pretraining-vitl:** five folds 9.1654 vs plain ViT-L 9.2330 (−0.068); CV +0.0078 strict [O-CV-SYN4-001]; CD −0.0031 strict, folds mixed [O-CD-SYN4-001]; PI: not in the CD stack [D-SYN-006].
- **idea-synthetic-pretraining-recipeA:** five folds 9.0198 vs plain recipe A s46 9.0261 (−0.006, CI −0.029..+0.016) [A-BB-SYN5-READ]; CD strict −0.0017 (CI −0.012..+0.008) [O-CD-SYN5-READ]; closed [D-SYN-008]. Fold-0 seed mean −0.022, which is smaller than the plain s47−s46 spread of +0.042 [A-BB-SYN7-RESULT].

Recipe A hyperparameter reads:
- **Encoder learning rate:** my lane never changed it with the joint loss. The CD lane's CD-only arms (folds 0+4 strict): 1e-5 −0.0030, 3e-5 +0.0181; both closed [O-CDTRAIN-1E5-F04, O-CDTRAIN-3E5-F04].
- **Drop path:** 0.2, closed (above).
- **Layer-wise decay:** tried only on ViT-L (above), never on recipe A.
- **EMA / SWA:** inside a 20-epoch recipe A run, swa6 equals raw (9.071 vs 9.068, fold 0). On ViT-L, swa6 beats raw (two folds 9.115 vs 9.231; synthetic ViT-L 9.0495 vs 9.1654).
- **Schedule:** CD is not falling at the last epoch [O-CDTRAIN-PROBE], so the longer schedule was dropped [D-CDTRAIN-002].
- **Augmentation:** geometric and C-Mixup are closed. Appearance augmentation was prepared but never run.
- **Seed noise:** recipe A s47 vs s46 +0.027 (CI +0.004..+0.047, folds 0+4).
- **Never tried on recipe A:** LLRD, weight decay, warm-up length, batch size, joint-loss weights (only the CD lane's CD-only form), whole-frame input resolution, more than 20 epochs with the joint loss, colour/appearance augmentation.

(c)
- **idea-appearance-augmentation:** `--appear P` in bb_runner (P=0 is pixel-identical to recipe A); never run, because it was closed together with shading.
- **idea-test-time-subspace-alignment:** not run; the coordinator's view was that a level offset cannot be fixed without labels.
- **idea-synthetic-pretraining-fulldata:** full-data synthetic ViT-L stopped at epoch 9 of 12 [D-SYN-006]; synthetic recipe A seeds 47 on folds 1-4 never run.
- **idea-residual-diversity-gate (L1):** residual r with recipe A of my members: synthetic recipe A 0.985-0.996, recipeA_win2x 0.93/0.97/0.98, CPSAM 512 0.89/0.93/0.95. Nothing below 0.9 on every metric was found.

## 5. Folder map

| Folder | Purpose | Tag | Reason / by-name references |
|---|---|---|---|
| `LB/source/` | runners, reads, synthetic code | KEEP | `bb_runner_attnpool_20261111.py` and `family_bb_20261108.py` are copied into the p/t1/t2/full build contexts at `code/experiments/lane_backbone_20261108/source/`. `lane_cellroute/inference/members.py:142` hard-codes `ATTN_SRC = .../lane_backbone_20261108/source`. A move breaks restaging and pod tests |
| `LB/runs/attnpool_f99_dinov3b16_s46/raw` | deployed HEX column | KEEP-MODEL | `manifests/deploy{,_p,_t1,_t2}.json`, `inference/params_build.py:16`, `docker/fp16_files.txt`, `package_20261112/{p,t1,t2}/model` hard links |
| `LB/runs/attnpool_f{0..4}_dinov3b16_s46` (predictions) | D-HEX-004 column OOF | KEEP | `lane_hex/source/joint_fit.py:37`, `joint_deploy.py:110`, `chain.py:254`. Weights: DELETE-AFTER-SUBMISSION |
| `LB/runs/vitl16_avg_f{0,4}_s46` | D-CD-001 ViT-L OOF folds 0/4 | KEEP (predictions + raw final.pt) | `cd_tiers_20261112.py:27`, `cd_read_20261108.py:45`, `cv_stack5_20261108.py:88`, `params_build.py:96`, `manifests/fold0.json`. Other tracks: DELETE-AFTER-SUBMISSION |
| `LB/runs/recipeA_win2x_*`, `cmix_*` | closed members read by lane scripts | ARCHIVE predictions; weights DELETE-AFTER-SUBMISSION | `cd_read_20261108.py:58`, `chain.py:252-253`, `cv_stack5_20261108.py:110` |
| other `LB/runs/<arm>` with checkpoints (synth*, vitl16_win2x, distill, tile, shift, shade, target, aux, droppath, geo, dp03) | closed arms | ARCHIVE predictions; weights DELETE-AFTER-SUBMISSION | `cd_synthft_avail`, `synthft_read`, `synthvitl_read`, `cd_tile_read`, `strict_joint_extra` scripts read some of them |
| suffix-less marker folders in `LB/runs/` (e.g. `synthft_ra_f0`, `synthpre3_f0`, `cmix_f0_g1`, `cdlane_*`, `cvlane_*`, `tile_cd_f0`, `_smoke_vitl`) | queue markers only (`queue_started`/`queue_exit.json`) | DELETE-NOW | no references |
| `LB/runs/*.lock`, `*.pid`, `LB/HOLD_*`, `LB/STOP_*`, `LB/queue_*.pid`, `LB/tmp/`, `LB/archive_holds_20261111/` | control files | DELETE-NOW | stale; nothing running |
| `LB/data/synth/synth_v3_frames.u8` | 63 GB synthetic set | DELETE-AFTER-SUBMISSION | all synthetic routes closed; seeded generator in source |
| `LB/data/synth/_archive_v1*`, `_archive_v2*` | rejected renders | DELETE-NOW | superseded |
| `LB/data/*.npz`, `LB/data/synth/synth_v3_labels.npz` | small inputs | ARCHIVE | `strict_joint_extra.py:10` reads `synth_zeroshot_f0pre_20261111.npz` |
| `LB/reads/`, `RESULTS.tsv`, `REPORT.md`, `protocol.json`, `logs/`, `jobs_*.tsv`, `NOTICE_*`, `HANDOFF_*` | evidence trail | ARCHIVE | ledger sources |
| `LB/archive/` | old runner copies; partial synthetic ViT-L f99 | ARCHIVE (code); DELETE-AFTER-SUBMISSION (the 1.2 GB ckpts) | none |
| `phase2_crossfold_20261015` | recipe A fold SSL5 folds 0/4, `source/` (recipe A code incl. `phase2_lejepa_20261003.py`), `data_preflight.json` | KEEP | `bb_runner`/`family_bb` paths; the image copies its source [F-DEPLOY-005] |
| `phase2_foundation_20261002` | `runtime/` (dinov3 + timm, image PYTHONPATH), `weights/dinov3-vitb16` | KEEP | Dockerfile ENV, `gc_entry.py` [A-DEPLOY-008/009]; `family_bb_20261108.py` |
| `phase2_backbone_screen_20261001/hf_cache` | timm ViT-L DINOv3 weights (PI-approved) | KEEP-MODEL | `family_bb_20261108.py`; ViT-L rebuilds |
| `phase2_pooling_20261020` | recipe A s46 fold 4 (`arms/std`) | KEEP | `read_bb_20261108.py` REF; recipe A OOF |
| `phase2_pooling_replication_20261026` | recipe A f0 s46/s47, f4 s47 (`*_std`); `gpu_keepalive.py` | KEEP | `read_bb_20261108.py` REF |
| other `phase2_*` (lejepa*, metric35_ssl, cpsam_scale, foundation_optimization/train, backbone_ssl_20261020, backbone_fold0_regression_ssl5_20261022) | not read by this lane | no call | purpose not in my records |
| `oof_recipeA_20261027` | recipe A seeds 46-48 OOF (s48 cut on folds 1-3), fold SSL5 folds 1-3, fold-2 s46_v2, `staging_regression_zero` | KEEP | `read_bb_ra5_20261112.py`, `synth_pretrain` SSL paths, `family_bb` |
| `fulldata_recipeA_20261028` | deployed recipe A s46/s47 epoch_20 (p and t3), fold-99 split, `ssl5_s1003` | KEEP + KEEP-MODEL | `manifests/deploy_p.json`/`deploy_t3.json` recipeA_tta; every f99 build |
| `recipe_A_20261027`, `queue_runner_20261027`, `domain_jepa_all`, `domain_jepa_small` | not read by this lane | no call | purpose not in my records |
| `ssl_screen_20260911`, `overnight_target_20260912`, `target_fold4_20260912` | hold the five-fold split files (`preflight/fold*.npz`) | KEEP (at least `preflight/`) | `bb_runner_20261108.py`, `family_bb_20261108.py` |
| `lane_gpu3_20261106` | CPSAM 512 f0/f4 OOF + SSL; plain DINOv3-B s46 f0/f4 (comparator for my DINOv3-B arms); `weights/dinov3-vitl16` | ARCHIVE; weights KEEP-MODEL | `family_bb_20261108.py` (dinov3-vitl16), `read_arm_vs_plain`, `read_distill` |
| `lane_gpu0_20261106`, `lane_gpu1_20261106`, `lane_gpu2_20261106` | not read by this lane | no call | purpose not in my records |
| `eval_harness_20261031` | metric, donors, bootstrap | KEEP | every read script |
| `meta_ceiling_20261103` | best pipeline (`best_pipeline_20261106.py`), `sweep/legible_window.npz` | KEEP | `read_bb_20261108.py --added`, `make_window_20261108.py` |
| `integration_rehearsal_20261110` | 11-10 rehearsal candidate OOF | ARCHIVE | `read_distill`, `read_arm_vs_plain`, decile read |
| `rootcause_20261109` | the 11-09 root-cause analysis cited in REPORT (detector-target headroom); matched by name, contents not read by this lane | ARCHIVE | none known |
| `/workspace/{artifacts,checkpoints,logs,scratch_*}` | not used by this lane | no call | not in my records |

## 6. Loose ends

- **O-BB-SYNFT3-5F was wrong. Superseded today by [O-BB-SYNFT3-5F-CORR].** That line's time_est reads 22:05; it was actually written at ~21:50 EST. The quoted "8.6736 as computed" is the 11-10 rehearsal OOF (`integration_rehearsal_20261110/out/integrated_oof.npz` P_main). Recomputed: 8.67356 pooled and 8.67357 as the per-fold mean. The frozen candidate is 8.66336 / 8.66867 strict [O-VER-002].
- **The same rehearsal OOF is the "candidate" in two other places.** It is behind the distillation "44 % of the gap" (RESULTS rows 52 and 57, 8.6876 on folds 0+4) and behind the label-decile bias read under F-SHIFT-001. None of my member numbers depend on it.
- **Two key names in the read JSONs are misleading.** `two_fold` is the mean over the listed folds, even with five folds. `a_plain_dinov3b_s46` in `reads/synthft_vitl_f01234_20261112.json` is the plain ViT-L comparator. Both names were inherited from the distillation read.
- **Seed means on folds 1-3 use two seeds only.** The recipe A s48 runs there were cut (`oof_recipeA_20261027/seeds_cut_log.json`), so seed means are s46+s47 on folds 1-3 and s46-48 on folds 0 and 4.
- **One fold-2 value is NaN.** The fold-2 HEX residual r in `synthft_ra_f01234_*` is NaN because of the two HEX-zero rows; scores skip them correctly.
- **Never run:** `LB/source/verify_synthvitl_f99_20261112.py` (D-SYN-006 cancelled the build), and the synth_v3 regeneration check.
- **Stale files to ignore:** `LB/runs/keepalive_gpu{0..3}.pid` name dead processes. The f0 ViT-L `ema9995` track is not for use.
- **The 11-09 rebuild sheet in REPORT.md is out of date in two places.** It lists a queued full-data recipe A win2x build that was cancelled (the arm was closed at CD five-fold +0.0026). Its CPSAM 512 note ("no fold checkpoints") still holds.
- **Note for the PI.** p and t3 load nothing else from this lane. If `LB/` is moved, restaging the p/t1/t2/full contexts needs `members.py:142` (`ATTN_SRC`) and the manifest paths updated. The p model tree itself holds hard links, so its files survive a move or deletion of the originals.
