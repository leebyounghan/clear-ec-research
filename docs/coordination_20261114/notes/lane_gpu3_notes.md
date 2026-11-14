# Lane GPU 3 (G4, model families), notes for sibling sessions

Started 2026-11-06 19:30 UTC. Files on the pod: `experiments/lane_gpu3_20261106/` (`protocol.json`, `jobs.json`, `queue.log`, `source/`, `runs/`, `reads/`, later `REPORT.md`).

- Keepalive 51428 stopped by pid. `source/queue_gpu3_20261106.py` (pid in `queue.pid`) runs `jobs.json` in order, re-reads it before each job, starts its own keepalive when the list runs out and stops it by pid before the next job.
- Runner `source/family_recipe_20261106.py`: recipe A budget on one GPU, only the encoder changes. Checked on CPDINO fold 0 seed 46: reference epoch 20 weights through its eval path return 9.1145 exactly; gradient norms at updates 1, 25, 50, 75, 100 are 0.564, 1.208, 1.135, 0.996, 4.156 against the reference 0.544, 1.227, 1.057, 1.002, 4.197 (RoPE coordinate jitter differs).
- Read `source/read_family_20261106.py`: same rule and added formula as lane GPU 0's `read_arm_20261106.py`, plus an added read per epoch with no checkpoint ensemble.

Measured on one H200, no keepalive, 35 GiB:

| Encoder | Micro-batch | Seconds per update | Seconds per SSL step | SSL + regression, one fold |
|---|---:|---:|---:|---:|
| CPDINO ViT-B, stride 8 | 4 | 2.9 | not measured | about 13 h |
| DINOv3 ViT-B/16, stride 16 | 16 | 0.39 | 3.1 | about 2.2 h |
| DINOv3 ViT-B/16, stride 8 | 4 | 2.9 | 16.4 | about 13.4 h |

## Update 2026-11-07 03:40 UTC

First read, DINOv3 ViT-B/16 stride 16, folds 0 and 4, seed 46: +0.274 (CI +0.185, +0.361) against recipe A; added to the best pipeline -0.0165 at w 0.125 (CI -0.028, -0.004), -0.012 at w 0.25 (CI crosses 0). Fails both gates. Details in the lane `REPORT.md` and `reads/dinov3_b16_s46.json`.

Measured cost, corrected: DINOv3 B/16 stride 16 is 1.56 h per fold, not 2.2 h. DINOv3 B/16 stride 8 SSL took 4.7 h; regression runs at 2.9 s per update.

Runner now also has `convnext_b32` (stride 32, grid 31x41, about 51 GiB at micro 16) and `dinov2_b14` (grid 70x93, about 38 GiB at micro 16); both passed a smoke run. Queue order: `dinov3_b16_s8` fold 0, `convnext_b32` folds 0 and 4, `dinov2_b14` folds 0 and 4, `dinov3_b16_s8` fold 4.

## Update 2026-11-07 03:55 UTC: GPU 3 handed to strong-view SSL

Pod-side `source/handover_20261107.sh` (log `logs/handover.log`) waits for `lane_gpu1_20261106/ssl/cropopt_both_f0/run/complete.json`, stops `dinov3_b16_s8` fold 0 by verified pid, and the queue then runs `source/e2e_cropopt_both_f0.sh`, a guarded call of lane GPU 1's `run_e2e.sh cropopt_both_f0 0 46`. Outputs go to `lane_gpu1_20261106/e2e/cropopt_both_f0/f0_s46/`. Lane GPU 1 must not queue this job.

Queue after it: `convnext_b32` folds 0 and 4, `dinov2_b14` folds 0 and 4, `dinov3_b16_s8` fold 0 from its saved SSL, then fold 4. That lasts past 12:00 EST even if the end-to-end job fails at once.

## Update 2026-11-07 08:00 EST: overnight loop ended

Reads so far, all seed 46, folds 0 and 4, against recipe A:

| Arm | Delta (CI) | Added, w 0.125 (CI) | Added, w 0.25 (CI) | Residual r with reference | Result |
|---|---:|---:|---:|---:|---|
| DINOv3 ViT-B/16, stride 16 | +0.274 (+0.185, +0.361) | -0.0165 (-0.028, -0.004) | -0.012 (-0.036, +0.012) | 0.90-0.95 | Fails both gates |
| Strong-view SSL `cropopt_both`, end to end | +0.0185 (-0.020, +0.058) | -0.0038 (-0.0097, +0.0017) | -0.0030 (-0.0151, +0.0081) | 0.98-0.99 | Fails both gates; lane GPU 1's verdict, same figures |

Queue on GPU 3: `convnext_b32` fold 0 in regression (194 s per epoch, SSL took 43 min, about 1.8 h per fold), then fold 4, then `dinov2_b14` folds 0 and 4. Estimated ends, Eastern: ConvNeXt 08:52 and 10:40, DINOv2 about 12:30 and 14:20, all 11-07. Keepalive starts by itself when the list runs out.

Lane GPU 2 runs `dinov3_b16` seed 47 and then `dinov3_b16_s8` folds 0 and 4 with this lane's runner and roots; the stride 8 read lands in this lane's `reads/` about 07:00 EST on 11-08.

## Update 2026-11-07 09:20 EST

ConvNeXt fold 0, one fold: +0.588 (CI +0.44, +0.74); added +0.005 at w 0.125. Fold 4 is running.

CPSAM on the whole frame is dropped on measured cost: 7.45 s per image, 98.9 GiB training and 59.7 GiB inference peak (`phase2_foundation_20261002/revision2/probes/cpsam/`). My earlier 45 h per fold was wrong. Runner now has `cpsam_c512`, central 512 crop, 22 GiB at micro 4, about 22 h per fold alone (estimate). Queued on GPU 3 after `dinov2_b14` fold 4.
