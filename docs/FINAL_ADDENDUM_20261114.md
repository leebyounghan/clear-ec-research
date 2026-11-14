# CLEAR-EC Phase II: addendum for 2026-11-14 (final model: tier p3b)

Written by the coordinator at 2026-11-14 15:56 EST. It continues `docs/FINAL_SUMMARY_20261113.md`; ledger ids are in brackets and the ledger is
`docs/coordination_20261114/ledger_20261111.jsonl`. MAPE in %, five folds, "strict" as defined in the 11-13 summary.

## 0. Bottom line

The final model is **tier p3b**. It is tier p with a shorter run time and with both ViT-L halves of the CD stack retrained at encoder
LR 2e-5. CV and HEX are tier p's.

| Tier | Change from the tier above | CD strict | Mean strict | Status |
|---|---|---:|---:|---|
| p | (11-13 build) | 6.1737 | 8.6682 | built, tested |
| p2 | DINOv3-B s47 CD half dropped; four read-once code switches | 6.1723 | 8.6677 | prepared on the pod, never built |
| p3 | ViT-L s46 replaced by the encoder-LR-2e-5 full-data build | 6.1608 | 8.6639 | built, tested, on the laptop |
| **p3b** | ViT-L s47 also replaced by its 2e-5 full-data build | **6.1479** | **8.6596** | built, tested, on the laptop |

- p3b against tier p, CD: strict -0.0258 (CI -0.040 to -0.012), repeated grouped 10-fold -0.0306 (CI -0.045 to -0.016); lower in every
  fold [O-CD-STAGE2-P3B, A-CD-P3B-CONSTANTS].
- p3b against p3, CD: strict -0.0129 (CI -0.022 to -0.004), 10-fold -0.0132 (CI -0.022 to -0.005); lower in every fold.
- Expected test score if the submission-1 gap between OOF and test repeats (CD -0.062, CV -0.061, HEX +0.395): about 8.750
  (CD 6.086, CV 10.051, HEX 10.114). This is an extrapolation from one submission, not a measurement.
- Leaderboard on 11-14 (PI): 1st 8.6920, 2nd 8.7035, 3rd 8.7264, 4th 8.7560; submission 1 is 7th at 8.9675.
- Upload, Try-out and the submission are the PI's actions. Their outcome is not recorded in this file.

## 1. What p3b is

- CD stack: log blend of P1 / recipe A / DINOv3-B s46b / ViT-L, deploy weights 0.25 / 0.40 / 0.05 / 0.30; mask-density slot
  (n_int >= 50, w 0.1); stretch g 1.075. Constants: `final_p3b_20261114/context/` (from
  `lane_cd_20261108/reads/freeze_cd_params_p3b_20261114.json`, md5 0442dd5a...).
- ViT-L slot V = geometric mean of two 9,000-row builds at encoder LR 2e-5 (plain runs use 1e-5), four-flip TTA at the final epoch:
  `arms/vitlfu_elr2e5_f99_s46` and `arms/vitl47fu_elr2e5_f99_s47`.
- Time: the read-once switches and the dropped DINOv3-B half bring the estimated T4 time per image from 287-314 s (tier p, two Colab
  VMs) to 258-296 s; the limit is 300 s. The platform time was not measured by us [O-DEPLOY-027].

## 2. Evidence for the ViT-L change

- Dose response from existing predictions: putting p3b's s46 half back to 1e-5 costs +0.0139 strict (CI +0.005 to +0.023) and
  +0.0151 on the 10-fold read; every fold is worse. Each seed half is worth about 0.012-0.013 and the two add up [O-CD-VITL-LR-DOSE-1E5].
- Member level, ViT-L s47 2e-5 against plain s47: -0.023 (CI -0.075 to +0.028). The stack gain is larger than the member gain.
- Full-data builds against their own fold family (CD, four-flip TTA): s46 2e-5 -0.29 %, s47 2e-5 +0.57 % (SE 0.11, 500 rows); the old
  s46 file in tier p was +1.25 % [O-CD-FAMILY-LEVEL, O-COORD-S47-LEVEL].
- 46 never-seen frames: no CD gap for the full-data builds [O-CD-FULLDATA-CHECKS].
- Pod parity, 5 frames: CV / HEX equal to p2 and p3 within 6.5e-9 relative; CD differs only through V [O-DEPLOY-P3B-PARITY].
- Box container test (RTX 4090, no network, 16 GB, uid 10001, one container per job): 17 of 17 jobs, 68.0-78.6 s per job on the
  forced T4 path; a garbage input returns the constants 2770 / 0.36 / 0.53 [O-DEPLOY-P3B-BOX-TEST].

## 3. Artifacts

| File | Bytes | sha256 |
|---|---:|---|
| `clear-ec_20261114-freeze-9687076-p3b.tar.gz` (image db390738df95) | 4,114,587,222 | 67a8452f62fbced75c6a3499e2c9b068ac0802e306b177381f73c154d3cd96c1 |
| `model_freeze-9687076-p3b_20261114.tar.gz` (443 files, 16,249,746,790 bytes inside) | 15,056,963,340 | a51a21581409b9cd7c527d90e8f4df6af67f268f575278a3583c15dfbbe3bf1d |

- Laptop: `~/Downloads/clear-ec-upload_20261114/`. The model archive was rebuilt there from the tier-p archive plus the two new
  ViT-L files and matches the box-tested tree file by file [O-COORD-P3B-LAPTOP-READY].
- Build box: `~/workspace/clear-ec/package_20261114/` (image file, patch folder, `test_model_p3b`).
- Try-out expected values, CD / CV / HEX (tolerance 0.1 % / 0.5 % / 0.5 %): 0001-19 2854.7 / 0.3819 / 0.5462; 0001-20
  1999.2 / 0.4796 / 0.4134; 0001-21 4373.7 / 0.2977 / 0.6496.
- The image and the model archive must be uploaded as a pair: an image of another tier with this archive returns the constants.

## 4. Closed on 11-14 (CD, against the stated control)

| Idea | Result | Ledger |
|---|---|---|
| DINOv3-B encoder LR 2e-5, five folds | stack -0.0032 strict, +0.0006 10-fold | O-COORD-DINO-2E5-FIVEFOLD |
| Recipe A encoder LR 2e-5 | stack +0.0015 / +0.0011, stopped at epoch 12 | D-CD-RECIPEA-2E5 |
| CD-weighted loss | member -0.071, stack +0.005 | O-COORD-CDW3 |
| Weight decay 0.05, LLRD 0.75, batch 32, drop-path, SWA | +0.040, +0.64, +0.10, +0.09, no gain | O-CD-NIGHT-CLOSE-20261114 |
| Source specialists | +0.305 on their own rows | O-COORD-SRCOLD |
| Trimmed loss | member +0.585 | O-COORD-PRL1 |
| Hard-image loss weight 2 on ViT-L 2e-5 | member +0.027 / +0.053, stack +0.0030 / +0.0026 (folds 0 / 4) | O-CD-HARD2-FINAL |

## 5. Not run, or stopped without a read

- ViT-L s46 encoder LR 3e-5: stopped on the PI's order at about 15:50 EST at about epoch 7 of 12 (folds 0 and 4) and epoch 1-2
  (folds 1 and 2). No read exists [D-PI-STOP-GPU-WRAPUP].
- P1 (five cpsam_v2 crop members, encoder LR 1e-5, stack weight 0.25) has never had an encoder-LR arm; it is the only CD slot where
  the increase that helped ViT-L is untested [O-CD-P1-LR-FACTS].
- Eleven training rows have CD errors above 100 % and contribute 0.263 to CD; six are both eyes of three cases and three differ from
  the prediction by almost exactly 2000 [O-COORD-11ROWS-FLAG]. Nothing was changed for them.

## 6. What this commit adds to the repository

- `final_p3b_20261114/context/`: the exact build context of the p3b image (Dockerfile, manifest, inference code, constants; no weights).
- `lanes_20261108/`: each lane's `source/`, `reads/`, `inference/` text files and its report, results and protocol files.
- `docs/coordination_20261114/`: the shared ledger, the deploy notes and the session notes.

## 7. Wrap-up actions on 11-14 (added 2026-11-14 16:04 EST)

- All GPU jobs were stopped on the PI's order at about 15:50 EST [D-PI-STOP-GPU-WRAPUP].
- Old packages were deleted on the laptop and the build box; p3b is kept on both and the t3 upload files are kept on the box until the
  score is known. One root-owned folder on the box (`train_20261113`, 1.6 GB) could not be removed [A-COORD-CLEANUP-20261114].
- The GPU pod `clear-ec-research-4gpu-1104` was deleted at 16:03 EST; its spec is saved and the volume `clear-ec-workspace` (experiments,
  data, checkpoints) is untouched. The CPU pod `clear-ec-cpu-1109` is still running [A-COORD-POD-DOWN-20261114].
