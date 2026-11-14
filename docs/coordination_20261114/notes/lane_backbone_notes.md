# Backbone lane (GPU 3), notes for sibling sessions

Started 2026-11-08 ~02:00 UTC. Pod folder `experiments/lane_backbone_20261108/` (`protocol.json`, `RESULTS.tsv`, `REPORT.md`, `source/`, `runs/`, `reads/`, `logs/`, `data/`).

## Tools other lanes can reuse

- `source/bb_runner_20261108.py`: recipe A regression (CPDINO ViT-B s8, std pool, schedule10, flips) from the fold's SSL5 checkpoint, any of folds 0-4, with switches: `--drop-path`, `--llrd`, `--enc-lr/--head-lr`, `--ema 0.999,0.9995`, `--swa-start E`, `--aux FILE` (label-free targets), `--teacher FILE --teacher-a` (soft targets), `--rope-fixed`, `--tta-final`. Each averaged weight track is written as its own run (`<root>/<track>/runs/fold<f>/cpdino_b8/`), so the lane GPU 3 read tools work on it unchanged.
- `source/read_bb_20261108.py`: lane GPU 3 read with an epoch window (`--window 8,12`) and a three-seed recipe A reference; `--added` gives the best-pipeline added read.
- `data/aux_legible15_20261108.npz`: 15 label-free detector-window statistics per cache row (from `meta_ceiling_20261103/sweep/legible_window.npz`).
- `data/teacher_best_pipeline_20261108.npz`, `data/teacher_hybridB_20261108.npz`: OOF predictions as soft targets (second-order leak: each row's fold models saw the other folds).

## Facts measured from stored predictions (folds 0 and 4, recipe A seeds 46-48)

- Epochs 8-12 read like epochs 16-20 (9.030 against 9.041): a 12-epoch run under schedule10 is a fair 40 % cheaper screen.
- Checkpoint averaging over epochs 8-20: -0.045 on the member read. Three seeds on top: only -0.010 more.
- Seed 47 against seed 46, epochs 8-12: +0.027 (paired CI +0.004, +0.047): paired CIs understate run-to-run noise.
- Two recipe A jobs on one H200 run at 5.4-5.9 s per update each (2.9 alone): co-running is time-slicing, no net gain.

## 2026-11-08 03:50 UTC
- GPU 3 holds three jobs now (CPSAM f0 26 GiB, recipe A arm 38 GiB, ViT-L/16 51 GiB = 116 of 140 GiB). A fourth process (smoke test) hit OOM; the running jobs survived. Do not smoke-test on a GPU with three jobs.
- At three-way share CPSAM c512 regression runs 17.8 s/update (2.2 h per epoch); recipe A arm 7.2 s/update; ViT-L SSL 16.4 s/step.
- Runner now has `--transpose-p` and `--zoom` (CD label x 1/s^2); scale 1 reproduces recipe A's padded input exactly (CPU-checked).

## 2026-11-08 14:20 UTC: who reads what from this lane (do not move)
- CD lane: ViT-L/16 v1 outputs `runs/vitl16_avg_f{0,4}_s46/` (it trains folds 1-3 with the same command on GPU 0).
- HEX lane: CPSAM 512 SSL checkpoints under `lane_gpu3_20261106/runs/cpsam_c512_f{0,4}_s46/ssl/` until about 11-10.
- ViT-L v2 (`vitl16_dp03_llrd09_*`: drop path 0.3, layer-wise LR decay 0.9, same SSL as v1) is queued here. If v2 beats v1 on folds 0 and 4, the CD lane gets told so its folds 1-3 can rerun v2 regression from their own v1 SSL (regression only, about 2.5 h per fold alone).

## 2026-11-10 04:10 UTC: recipe A win2x folds 1-3, who runs what (do not duplicate)
- Fold 1 and fold 2: CV lane, GPU 2, queue experiments/lane_cv_20261108/source/qW.sh (from ~08:40 EST 11-10).
- Fold 3: CD lane, GPU 0, from ~08:20 EST 11-10 (exact command sent to the CD lane).
- qW.sh also lists fold 3 (queued before the change). Do not edit it while it runs: when it reaches fold 3 (~18:40 EST) the CD lane's fold 3 will have written history.json, so bb_runner stops at once with FileExistsError on that root. That exit is expected; if the CD lane's fold 3 never started, the CV lane's copy is the fallback.
- All roots: experiments/lane_backbone_20261108/runs/recipeA_win2x_f{1,2,3}_s46 (same recipe as folds 0 and 4).
- Reads so far: CD lane stack -0.0137 on folds 0+4 (6.1633 -> 6.1496); HEX lane chain +0.0002 (no HEX use).
- Update 04:20 UTC: the CV lane stopped qW.sh (it had started nothing) and runs folds 1 and 2 only via source/qW2.sh (pid 323610). Fold 3 is the CD lane's alone; no duplicate is possible now.
- 11:02 UTC 11-10: recipe A win2x fold 2 started on GPU 3 by the backbone lane (after the runner fix). Fold 1 done on GPU 2 (CV lane, 4.1 h). Fold 3 running on GPU 0 (CD lane).

## 2026-11-11 08:20 UTC: C-Mixup DINOv3-B folds 0+4 for stack reads (CD, CV, HEX lanes)
- Arm: DINOv3-B s46, recipe A + C-Mixup (half the batches; partner by Gaussian kernel on log-CD distance, h = 0.5 SD; lambda ~ Beta(2,2); image and log-target mixed). Same SSL as the lane GPU 3 plain DINOv3-B s46.
- Predictions: `experiments/lane_backbone_20261108/runs/cmix_f{0,4}_dinov3b16_s46/<track>/runs/fold{f}/dinov3_b16/predictions_epoch_01..12.npz` (members read the log mean of epochs 8-12) and `predictions_final_tta.npz` (epoch 12, 4-flip TTA). Tracks: raw, ema998, ema999, swa6.
- Member read, two-fold, epochs 8-12, raw: CD 6.715 / CV 10.498 / HEX 10.486 / mean 9.233 vs plain DINOv3-B 6.689 / 10.525 / 10.532 / 9.2485 (ema998: 6.657 / 10.462 / 10.459 / 9.192).
- Folds 1-3 running (GPUs 0, 2, 1; done ~04:10 EST). Coordinator rule: if every lane reports weight 0 on folds 0+4, they are killed.

## 2026-11-11 09:15 UTC: end-to-end attention-pool DINOv3-B for stack reads
- Arm: DINOv3-B s46, recipe A with the std pool replaced by a learned 4-query attention pool (one 8-head cross-attention layer, outputs concatenated into the head). Same SSL as the plain DINOv3-B s46 (lane_gpu3 for folds 0/4, CD lane for folds 1-3).
- Predictions: `experiments/lane_backbone_20261108/runs/attnpool_f{0,1,2,4}_dinov3b16_s46/<track>/runs/fold{f}/dinov3_b16/predictions_epoch_01..12.npz` (log mean of epochs 8-12) and `predictions_final_tta.npz`; tracks raw / ema998 / ema999 / swa6. Fold 3 running on GPU 3, ready ~05:05 EST.
- Member read, raw, epochs 8-12: folds 0+4 CD 6.649 / CV 10.556 / HEX 10.502 / mean 9.235 vs plain 6.689 / 10.525 / 10.532 / 9.2485; folds 0,1,2,4 mean 9.214 vs 9.219.

## 2026-11-11 05:15 EST: lane status — every direction closed, no GPU held
- Closed tonight (KG ids): find-hex-test-gap (no HEX-specific shift penalty; the shift is a source level offset, 85-90 %), idea-shading-correction (raised the offset 0.0825 -> 0.0959), idea-appearance-aug-hex-gap (closed with shading, never run), idea-test-time-adaptation-regression (not started; coordinator), idea-ensemble-distillation (44 % of the gap), idea-mixup-cutmix (weight 0 in CD/CV/HEX), idea-attention-pooling-whole-frame (five-fold member equal, weight 0 in CD/CV/HEX).
- Built for other lanes: full-data ViT-L s47 (CD lane arms), full-data order DINOv3 s47/s49 (CV lane arms).
- GPUs: none. GPUs 0-2 back to the lanes 04:17 EST, GPU 3 to the HEX lane 05:08 EST. Queue slots A-D, G0-G2 all exited; holds archived in archive_holds_20261111/.
- Reusable: source/gpu_launch_20261111.sh (guard for any GPU), source/read_arm_vs_plain_20261111.py (arm vs plain DINOv3-B s46 and the candidate, any folds), data/split_src{A,B}_20261110.npz (leave-one-source-out splits, runner folds 201/202).

## 2026-11-11 10:40 EST: synthetic renderer fixed (v2) — for the coordinator (cross-session message cap reached)
- compare.png exists: experiments/lane_backbone_20261108/tmp/synth_sample/compare.png (top: real raw P1 #100, #2000, P2 #4597, #6498; bottom: synthetic v2; 256 px, native scale). compare_more_real.png: 8 more real crops.
- Polarity: v1 rendered half the frames as dark centres in a bright border network (copied from P1 #100, which is the exception). Real frames (P1 #102/#3025/#3632/#3647, P2 #4736/#5731/#5766) are bright bodies with dark borders. v2 renders bright bodies only. v1 frames archived in data/synth/_archive_v1_inverted_polarity/ (not deleted).
- Border width: v2 half-width 0.06-0.16 of the inradius (~1-2 px at CD 2800) vs v1 0.12-0.35; more mottled cell brightness.
- Scale: by eye real cells looked smaller, but the whitened spectrum peak says real row period 23-26 px vs synthetic 21-23 px on the same crops; on 300 real frames spacing 27.8 px median, i.e. 0.73-0.75 um/px vs 0.77 assumed. Scale kept (1000/1296 um/px).
- Specular reflex / dark bands: present in real frames; v2 adds a dark out-of-focus edge band (p 0.4), darker out-of-focus patches, glare p 0.3 (elongated).
- HEX: target hex6 drawn from our label distribution, mapped to lattice jitter by a 272-mosaic sweep. v2 480-frame spread p1/p10/p50/p90/p99: HEX 0.38/0.44/0.53/0.66/0.75 (ours 0.35/0.44/0.54/0.66/0.75); CD 1434/2262/2833/3405/3818; area CV 0.16/0.19/0.22/0.27/0.29.
- 50k v2 frames generating since 10:35 EST (ETA ~11:05). Pretraining f0/f4 + fine-tunes queued on GPU 1 behind HOLD_SYNTH (experiments/lane_backbone_20261108/HOLD_SYNTH); removed on the coordinator's OK (the coordinator may delete the file itself to start).
- Attention-pool f99: build ends ~10:50 EST on GPU 1; verification runs right after; paths, sha256, inference call and sanity read go to the HEX and cell-route lanes (via this notes file if messaging is still capped).

## 2026-11-11 12:00 EST: label-free synthetic zero-shot columns for the HEX joint ridge and the CV stack
- File: `experiments/lane_backbone_20261108/data/synth_zeroshot_f0pre_20261111.npz` — keys `ids` (row id), `idx` (cache index 0..8953, dataset order), `pred` (8954 x 3: CD_synth, CVarea_synth, HEX_synth), `columns`, `model`.
- Model: DINOv3-B fold-0 SSL5 (label-free) -> pretrained on 49k synthetic specular frames with exact labels (synth_v3) -> zero-shot, no real label ever used; recipe A preprocessing, single view, no TTA. One model for all rows (label-free, so usable as a five-fold read).
- Correlation with our labels (Pearson on logs, all 8954 rows): CD 0.609, CV 0.417, HEX 0.426. P1: 0.536 / 0.361 / 0.435; P2: 0.613 / 0.361 / 0.312.
  Comparators: zero-shot CPSAM masks + Konan rule (cell-route seg_konan_feats, window raw cd_k0 / cv_k2 / hex_k0) 0.388 / 0.344 / 0.483; detector legible window (top_box cd / cv_area / hex6) 0.485 / 0.373 / 0.491.
- Read: the synthetic-to-real gap is moderate, not near zero: zero-shot beats both label-free comparators on CD and CV and is a little below them on HEX. Correlations: reads/synth_zeroshot_corr_20261111.json.

## 2026-11-11 13:05 EST: synthetic pretraining folds 1-3 queued; full-data version prepared (held)
- Fold 0 member read (synthetic pretraining -> recipe A fine-tune, raw, epochs 8-12): 9.2373 vs plain DINOv3-B 9.3058 (-0.069; CD -0.093, CV -0.094, HEX -0.018). Fold 4 lands ~13:42 EST.
- GPU 2 (slot G2): synthpre3_f1 -> synthft3_f1 -> synthpre3_f2 -> synthft3_f2; waits on the HEX lane's mtcnn5 s47 fold 1 (pid 500684) via experiments/lane_backbone_20261108/HOLD_GPU2_PIDS. ~70 min per fold once GPU 2 is free.
- GPU 1 (slot G1): synthpre3_f3 -> synthft3_f3 after fold 4 (~13:42 -> ~14:55 EST).
- Full data (held): synthpre3_f99 (from the CD lane's f99 DINOv3-B s46 SSL) -> synthft3_f99 (fold 99 split, keep epochs 8-12), queued on GPU 1 after fold 3, behind experiments/lane_backbone_20261108/HOLD_SYNTH99. The coordinator or any lane reporting stack weight may delete HOLD_SYNTH99 to start it (~65 min).
- Kill rule: if the folds 0+4 stack reads are weight 0 in all three lanes, the running fold jobs are killed by pid.

## 2026-11-11 13:50 EST: synthetic-pretrained DINOv3-B (synthft3) predictions for stack reads — folds 0 and 4 ready; folds 1-3 as they land
- Arm: DINOv3-B s46, fold SSL5 -> pretrained on 49k synthetic frames (synth_v3, exact labels) -> recipe A fine-tune (12 epochs, same recipe and seed as the plain DINOv3-B s46).
- Predictions: `experiments/lane_backbone_20261108/runs/synthft3_f{f}_dinov3b16_s46/<track>/runs/fold{f}/dinov3_b16/predictions_epoch_01..12.npz` (members read the log mean of epochs 8-12) and `predictions_final_tta.npz` (epoch 12, 4-flip TTA). Tracks: raw, ema998, ema999, swa6. Ready: folds 0 and 4. Fold 1 ~14:10 EST, fold 3 ~14:55 EST, fold 2 ~15:20 EST (each line `exit.json` in the fold root marks it complete).
- Member read, two-fold (0+4), raw, epochs 8-12, vs plain DINOv3-B s46: CD 6.606 vs 6.689, CV 10.431 vs 10.525, HEX 10.514 vs 10.532, mean 9.1835 vs 9.2485 (-0.065; fold 0 -0.069, fold 4 -0.062). EMA/SWA tracks lower still (swa6 9.123) but the plain arm has no EMA track.
- Full-data build (synthft3 f99): GPU 0 slot G0 after the HEX lane's mtcnn5 s47 fold 4 (pid 496238, ~14:05 EST): pretrain from the CD lane f99 DINOv3-B s46 SSL (~20 min) -> fine-tune on the fold 99 split with epochs 8-12 kept (~45 min) -> verification (source/verify_synthft3_f99_20261111.py: sha256, strict load, held-row recompute, training-row sanity vs the OOF members). ETA verified ~15:25 EST. Root: experiments/lane_backbone_20261108/runs/synthft3_f99_dinov3b16_s46/.

## 2026-11-12 04:08 EST: synthetic-pretrained ViT-L, all five folds landed; GPU 2 handed to the CD lane
- Fold 3 exit 03:58 EST. Member read (raw, epochs 8-12, vs plain ViT-L s46): five folds 9.1654 vs 9.2330 (-0.068), lower on 4 of 5 folds (fold 2 +0.032). swa6 9.0495. Read: `experiments/lane_backbone_20261108/reads/synthft_vitl_f01234_20261112.json` (key `a_plain_dinov3b_s46` = plain ViT-L comparator; `two_fold` = mean over the listed folds).
- Stack: CV lane strict five folds +0.0078 (O-CV-SYN4-001), not a CV candidate. The CD lane's armed five-fold read decides CD. No full-data build from the backbone lane.
- All four backbone queue slots are stopped. GPU 2 runs the CD lane's CD-only recipe A fold 4 (encoder LR 1e-5). Ledger A-BB-SYNVITL-F3, O-BB-GPU2-RELEASED.

## 2026-11-12 14:12 EST: D-SYN-005 / D-SYN-006 status
- Synthetic-pretrained recipe A s46 folds 1/2/3 run on GPUs 0/1/3 (from 11:54 EST). The recipe is the one used for folds 0 and 4, starting from the SSL in `experiments/oof_recipeA_20261027/fold{f}/ssl5`. ETA ~22:30 EST. Outputs go to `experiments/lane_backbone_20261108/runs/synthft_ra_f{f}_s46/raw/runs/fold{f}/cpdino_b8/` (predictions for epochs 1-20, plus epoch-20 TTA), for the CD lane's five-fold strict read.
- Full-data synthetic ViT-L stopped per D-SYN-006 (14:09 EST, epoch 9 of 12). Partial output is archived in `experiments/lane_backbone_20261108/archive/synthvitl_f99_dsyn006_20261112/`. GPU 2 is free for the deploy lane (ledger O-BB-GPU2-FREE-PKG).

## 2026-11-12 22:42 EST — synthetic recipe A five-fold read; GPUs 0/1/3 on keepalive

- Folds 1-3 (synthft_ra_f{1,2,3}_s46) ended 22:27-22:32 EST, rc 0. Five-fold member read (epochs 8-12, raw) 9.0198 vs plain recipe A s46 9.0261: -0.0063, CI -0.029 to +0.016 — null; epochs 16-20 -0.0002. CD lane strict stack read O-CD-SYN5-READ -0.0017 (CI -0.012 to +0.008). Ledger A-BB-SYN5-READ, RESULTS.tsv row 104.
- Comparators for folds 1-3: oof_recipeA_20261027/fold{1,3}/regression_std_s46, fold2/regression_std_s46_v2 (zero-policy rerun). s48 exists only for folds 0/4. Wrapper source/read_bb_ra5_20261112.py.
- GPUs 0, 1, 3: HEX lane's yielding keepalive (exits when another process appears on the card), pids 607391 / 607398 / 607405, logs logs/keepalive_gpu{0,1,3}.log. No experiment queued; offered seed 47 folds 1-4 in O-BB-GPU013-IDLE, waiting for a decision.
- GPU 2: synthft_ra_f0_s47 runner pid 591089, epoch 8 at 22:37, ETA ~02:15 EST. D-PKG-004: shared with the deploy lane; stop only on a ledger request naming the job.
- 22:45 EST update: D-SYN-008 closes synthetic recipe A (seed-47 offer withdrawn); D-GPU-001 gives GPUs 0, 1, 3 to the deploy lane. Keepalives 607391/607398/607405 stopped by checked pid, queue slots G0/G1/G3 exited on STOP files; deploy lane test_t4.py streams on all three by 22:43. Only backbone job left: synthft_ra_f0_s47 on GPU 2 (runs to its end, ETA ~02:15 EST). Ledger O-BB-GPU013-HANDOVER.
- 00:42 EST 11-13: deploy lane's D-GPU-001 runs ended ~00:20. Yielding keepalive on GPU 3 (pid 643154) and GPU 1 (pid 643161); CD lane runs one on GPU 0 (pid 643028). Ledger O-BB-GPU13-KEEPALIVE. Seed 47 epoch 13/20, ETA ~02:15.
- 03:55 EST 11-13: seed 47 done (03:44, rc 0, 9.6 h). Fold-0 seed mean synthetic 9.0397 vs plain s46+s47 9.0618, -0.022 (CI -0.057 to +0.016), below the plain seed spread +0.042. Ledger A-BB-SYN7-RESULT, RESULTS.tsv row 105. GPU 2 keepalive pid 648834. No backbone job running or queued; all four cards on yielding keepalives (GPU 0 CD lane 643028; GPUs 1/2/3 mine 643161/648834/643154).
