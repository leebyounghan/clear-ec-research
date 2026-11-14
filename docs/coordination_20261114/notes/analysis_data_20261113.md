# CLEAR-EC final candidate: data handling (rows, augmentation, preprocessing)

Written 2026-11-13. Read-only analysis of the pod tree `/workspace/clear-ec-research` (pod `clear-ec-research-4gpu-1104`) and the local lane summaries. Paths are relative to that tree unless they start with `/`. `E/` = `experiments/`. Counts were recomputed from the files; the snippet behind each count is in section 5. Measured facts only.

Final candidate = tier p of the 2026-11-12 freeze (`E/lane_cellroute_20261109/inference/manifests/deploy_p.json`, `params/deploy_p.json`; `final_summary_deploy_20261113.md` section 2).

## 0. Short answers

- **The gap to 9,000 was created by us, not by the download.** The archive holds 9,000 `.mha` files and the label CSV holds 9,000 rows with 9,000 unique IDs. The cache builder kept a file only when its name equalled the CSV ID exactly (`code/cache_native.py:7`). 46 CSV IDs differ from their file names by a trailing space, a hyphen, a double space or a missing space, so 46 labelled, readable frames were left out: 9,000 - 46 = 8,954.
- **Zero labels: 2 rows, HEX only.** `2019-0064-ODCN` (CD 1053, CV 0.03, HEX 0) and `2019-0064-OSCN` (CD 520, CV 0.11, HEX 0), the two eyes of one case, both in fold 2. No zero in CD or CV; no NaN or negative label.
- **Scored rows (OOF): CD 8,954, CV 8,954, HEX 8,952.** Trained rows of the full-data regression models: 8,952 (8,954 minus the two HEX-zero rows). The five Phase-1 crop members in the package were trained on 8,998 rows (all 9,000 reconciled rows minus the two).
- **"8953" matches no file.** No table or array has 8,953 rows. It appears only in prose (`docs/BRIEF.md:14,31,40`, the docstring `E/meta_ceiling_20261103/source/best_pipeline_20261106.py:2`, the wiki page name). `E/integration_rehearsal_20261110/REHEARSAL.md:39` already records this.
- **"8825" = 8,954 - 20 - 109.** Cell-route ridge fit rows: minus the 20 frames whose expert markers trained the detectors, minus 109 rows without usable cell-route features (the two HEX-zero rows are among the 109).
- **No line or edge enhancement exists anywhere.** The only contrast operation is CLAHE followed by 1-99 percentile scaling. Flat-field correction was tested and closed; sharpening and denoising were never used.

## 1. Row accounting

### 1.1 Waterfall

| step | rows | dropped | which rows, why | code / file |
|---|---|---|---|---|
| Challenge description | 9,000 | | | challenge page |
| `.mha` files in the archive `clear-ec-train9000-raw.tar.gz` | 9,000 | 0 | 8,954 are in the 972x1296 uint8 memmap; the other 46 were re-read: 972x1296 uint8, readable | `datasets/clear_ec_fullframe_20260904/fullframe_manifest.json` (`raw_file_count` 9000); `ls train_mha \| wc -l` = 9000 |
| Label rows in `final_train_ids.csv` | 9,000 | 0 | 9,000 unique IDs, `Image Number` = 1 on every row, no NaN, no negative | `datasets/clear_ec_official/final_train_ids.csv`; `raw_id_alias_audit.json` (`label_count` 9000) |
| Cache rows (file name == CSV ID, exact string) | 8,954 | 46 | ID spelling mismatch between file name and CSV (list in 1.2). Not missing images, not missing labels | `code/cache_native.py:4-7`; `fullframe_manifest.json` (`matched_cache_rows` 8954, `extra_raw_ids` 46) |
| Full-frame memmap `img_full_raw.dat` | 8,954 | 0 | same rows, same order as the cache; pixel sha256 verified | `fullframe_manifest.json`; `E/phase2_crossfold_20261015/data_preflight.json` (`pixels.verified` true, `mismatch_rows` 0) |
| OOF table (every lane's `idx`, `y`, `fold`) | 8,954 | 0 | folds 1791 / 1791 / 1791 / 1791 / 1790 | `E/recipeA_analysis_20261028/hybrid_oof.npz` via `E/eval_harness_20261031/source/eval_oof_20261031.py:22-26` |
| Scored rows, CD | 8,954 | 0 | | `eval_oof_20261031.py:42-50` (`ape`: `y > 0`, `nanmean`) |
| Scored rows, CV | 8,954 | 0 | | same |
| Scored rows, HEX | 8,952 | 2 | HEX label 0: `2019-0064-ODCN`, `2019-0064-OSCN` (fold 2 scores 1,789 HEX rows) | same |

### 1.2 The 46 rows left out of the 8,954

All 46 are new-format IDs (2019-2022), all have positive labels, all files read as 972x1296 uint8, none is pixel-identical to a cache row or to another of the 46. They were reconciled on 2026-09-03 (`raw_id_alias_audit.json`, rule: remove whitespace/hyphens only for unmatched IDs, unique keys, no fuzzy matching) and are used only by the Phase-1 all-data fits (1.4).

| mismatch | n | file names |
|---|---|---|
| CSV ID has a trailing space | 33 | `2019-0040 ODCN`, `2019-0081 OSCN`, `2019-0443 OSCN`, `2019-0469 OSCN`, `2019-0602 OSCN`, `2019-0717 OSCN`, `2020-0197 OSCN`, `2020-0302 OSCN`, `2020-0601 ODCN`, `2020-0693 ODCN`, `2020-0862 OSCN`, `2020-0881 OSCN`, `2020-0907 ODCN`, `2021-0034 OSCN`, `2021-0067 ODCN`, `2021-0306 OSCN`, `2021-0360 ODCN`, `2021-0446 ODCN`, `2021-0449 ODCN`, `2021-0725 ODCN`, `2021-0735 ODCN`, `2021-0772 OSCN`, `2021-0808 ODCN`, `2021-0933 ODCN`, `2022-0029 ODCN`, `2022-0048 ODCN`, `2022-0059 ODCN`, `2022-0416 OSCN`, `2022-0448 ODCN`, `2022-0497 OSCN`, `2022-0518 OSCN`, `2022-0720 ODCN`, `2022-0751 ODCN` |
| CSV has a hyphen, file has a space | 6 | `2019-0269 ODCN`, `2019-0269 OSCN`, `2019-0596 OSCN`, `2019-0728 OSCN`, `2020-0042 ODCN`, `2020-0042 OSCN` |
| CSV has a space, file has a hyphen | 4 | `2019-0333-ODCN`, `2019-0333-OSCN`, `2019-0701-OSCN`, `2019-0727-OSCN` |
| CSV has a space, file has "- " | 1 | `2021-0934- ODCN` |
| CSV has no separator | 1 | `2021-0911 ODCN` (CSV `2021-0911ODCN`) |
| CSV has a double space | 1 | `2022-0033 OSCN` |

### 1.3 Label and ID statistics

- **Zero labels:** CD 0, CV 0, HEX 2 (both in the 8,954 and in the 9,000). Label ranges in the 9,000: CD 372-4739, CV 0.03-2.49, HEX 0.00-0.90.
- **Duplicated IDs:** none (9,000 unique).
- **ID formats:**
  - Old `NNNN-YY`: 4,514 rows in the CSV and in the cache (year suffix 18: 292, 19: 786, 20: 506, 21: 795, 22: 694, 23: 705, 24: 736).
  - New `YYYY-NNNN O[DS]CN`: 4,486 rows in the CSV, 4,440 in the cache (2019: 1196, 2020: 999, 2021: 1099, 2022: 1192 in the CSV).
- **Donor groups (`slide_id`) in the 8,954:** 6,768. Every old-format row has `slide_id` = ID, so the 4,514 old rows are 4,514 singleton groups. The 4,440 new-format rows form 2,254 groups (69 singletons, 2,184 pairs, 1 triple).
- **Fold assignment:** `code/train_oof_h200.py:116-131` (`make_group_folds`). Groups by `slide_id`, shuffled with seed 2026, sorted by size, each group to the currently smallest fold. No `slide_id` spans two folds (checked). Every later lane reuses these folds.
- **Pixel-identical frames under different IDs:** 14 pairs (28 rows) in the 8,954 (`E/ssl_screen_20260911/preflight/pixel_fingerprints.npz`).
  - 3 pairs are one slide under two ID spellings with identical labels: `2019-0247 OSCN` / `2019-0247-OSCN`, `2019-0532 OSCN` / `2019-0532-OSCN`, `2019-0357 ODCN` / `2019-0357-ODCN`.
  - 1 pair is the OD and OS rows of one slide with the same pixels and different labels: `2019-0271-ODCN` / `2019-0271-OSCN`.
  - 10 pairs are different slide IDs with the same pixels and different labels, and each pair sits in two different folds: `2021-0328 OSCN` / `2021-0348 OSCN`, `2021-0325 ODCN` / `2021-0345 ODCN`, `2021-0716 ODCN` / `2021-0717 ODCN`, `2022-0261 ODCN` / `2022-0361 ODCN`, `2022-0365 OSCN` / `2022-0635 OSCN`, `2022-0365 ODCN` / `2022-0635 ODCN`, `2022-0875 ODCN` / `2022-0895 ODCN`, `2020-0314 OSCN` / `2020-0344 OSCN`, `2021-0694 OSCN` / `2021-0894 OSCN`, `2020-0721 OSCN` / `2020-0731 OSCN`.
  - No row was removed for being a duplicate in any full-data fit. All 28 rows are scored.

### 1.4 Rows used for training, per family

| family (package folder) | fold fits (OOF), train rows per fold 0-4 | full-data fit in the package | rule | code |
|---|---|---|---|---|
| Phase-1 crop members: seed42, seed77, seed2026, metric35, predbag2_35 (`E/deploy_review_20260905/staged/model`) | 7161 / 7161 / 7163 / 7161 / 7162 | **8,998** of the 9,000 reconciled rows | Other folds, minus rows with any label <= 0. These fold fits keep the train rows linked to validation by pixel-identical frames. The full-data fit uses the separate 9,000-row cache, so it includes the 46 rows of 1.2 | `code/train_oof_h200.py:139`; `code/prepare_final9000_candidate.py:47-54`; `code/train_final_candidate.py:34`; `code/train_deploy_methods.py:74-78`; `datasets/clear_ec_final9000_20260904/manifest.json` |
| Recipe A (CPDINO ViT-B), seeds 46/47 (`E/fulldata_recipeA_20261028`) | 7151 / 7157 / 7153 / 7161 / 7150 | **8,952** | Cleaned split files: Phase-1 train rows minus rows whose slide is linked to a validation row through a pixel-identical frame (10 / 4 / 10 / 0 / 12 rows). Full data: 8,954 minus the two HEX-zero rows | splits listed in `E/lane_backbone_20261108/source/bb_runner_20261108.py:42-44`; `E/meta_ceiling_20261103/source/recipeA_fulldata_regression_20261028.py:49-65`; `E/fulldata_recipeA_20261028/fulldata_regression_addendum.json`; `E/phase2_crossfold_20261015/data_preflight.json` (`legacy_cleaning`) |
| DINOv3-B/16 (CD and CV term), ViT-L/16, attention-pool DINOv3-B, CV-weighted DINOv3-B (`E/lane_cd_20261108/arms/*_f99_*`, `E/lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46`, `E/lane_hex_20261108/deploy/cvw3_f99_s46,_s47`) | same split files as recipe A | **8,952** | fold 99 = `fulldata_regression_split.npz` (train 8,952, held 2) | `E/lane_cd_20261108/source/family_cd_20261108.py:30-33`; `bb_runner_cd_20261108.py:34-37`; `E/lane_backbone_20261108/source/bb_runner_attnpool_20261111.py:53` |
| SSL5 pre-training of the encoders | fold train rows | recipe A: 8,954 images (labels unused); DINOv3-B and ViT-L: 8,952 per the CD summary | | `E/fulldata_recipeA_20261028/ssl5_s1003/manifest.json` (`train_n` 8954); `final_summary_cd_20261113.md:59` |
| Point detectors `sg2_*` (13 non-pre-only detectors and hex6) | per fold | **20 frames** with expert cell markers ("train-green": `0323-23, 0324-23, 0326-23, 0329-23, 0330-23, 0334-23, 0335-23, 0341-23, 0342-23, 0344-23, 0348-23, 0367-23, 0379-23, 0380-23, 0385-23, 0388-23, 0580-23, 0582-23, 0585-23, 0586-23`); 5 more marker frames are validation only | encoder initialised from the recipe A fold or full-data encoder | `E/green_track_20261028/source/green_sg2_specialist_20261028.py:1-10`; `E/green_point_probe_20261023/data_binding.json` |
| Pre-only detectors `sg2pre_s46/47/48` | per fold | **307 external images** (Topcon SREP set, 385 minus images with < 10 points and listed duplicates); no CLEAR-EC marker frame | | `E/green_track_20261028/deploy/detectors/sg2pre_s46.json` (`n_images` 307); `source/green_external_srep_20261030.py:5-19` |
| Cellpose-SAM segmentation (`cpsam_models/cpsam_v2`) | not trained | not trained (zero-shot) | | `E/lane_cellroute_20261109/inference/recipe_infer.py:48-57` |
| Cell-graph networks (CV and HEX) | other folds & `good` & non-empty graph | all `good` rows with a non-empty graph; count not computed | `good` = 8,934 | `E/lane_cv_20261108/source/graph_train_cv_20261108.py:17,78` |

### 1.5 Reduced subsets in the stacking steps

`good` = not one of the 20 train-green frames = **8,934 rows** (`best_pipeline_20261106.py:12`, flag `seen_leaked` in `E/submission2_search_20261103/green_features.npz`). The 20 frames are still predicted and scored; they are only kept out of every fit that reads detector outputs. Fold counts of the 20: 6 / 5 / 4 / 5 / 0.

| step | rows | how the number is built | code |
|---|---|---|---|
| Stretch and stack fits (CD, CV, HEX) | 8,934 (`good`), per fold `(fold != f) & good` | 8,954 - 20 | `best_pipeline_20261106.py:22`; `E/lane_cd_20261108/source/cd_common.py:31-34`; `E/lane_cv_20261108/source/cv_deploy_fit_20261110.py:95`; `E/lane_hex_20261108/source/joint_fit.py:57,73` |
| hex6 feature used | 8,927 | 8,947 rows with a valid feature, minus the 20 (7 `good` rows have no feature and keep the base HEX) | `best_pipeline_20261106.py:12-13` |
| Mask-density CD slot applied (`n_int >= 50`) | 7,704 | 8,954 - 535 rows with no mask features - 715 rows with fewer than 50 interior cells. Calibration fit rows over all folds: 7,688 | `E/lane_cd_20261108/source/cd_combine_20261110.py:14-20`; `cd_freeze_20261111.py:25` |
| Cell-route HEX ridge, training rows | **8,825** | 8,954 - 20 train-green - 109 without usable features (71 non-finite, 38 with fewer than 50 full-frame cells; the 2 HEX-zero rows are among the 109). Per fold: 7070 / 7065 / 7055 / 7060 / 7050 | `E/lane_cellroute_20261109/source/deploy_hexkonan.py:52-58`; `corr_newbase.py:13-16`; `inference/params/cellroute_deploy.json` (`train_rows` 8825) |
| Cell-route correction applied | 8,845 | rows with usable features; the other 109 keep the base HEX (77 old-format, 32 new-format) | `deploy_hexkonan.py:61-64` |

No outlier filter, label clipping or image-quality filter exists in any step read.

## 2. Augmentation

### 2.1 Train-time augmentation of the adopted runs

| family | train-time augmentation | code |
|---|---|---|
| Phase-1 crop members (Cellpose-SAM encoder, mean pool) | Random 256x256 crop anywhere inside the central 512x512 (offsets uniform 0-256); horizontal flip p 0.5; vertical flip p 0.5; rotation by k x 90 deg, k uniform in 0-3. `predbag2_35` draws 2 such crops per image and averages the predictions. No intensity augmentation | `code/train_oof_h200.py:176-186`; staged copy `E/deploy_review_20260905/staged/final_candidate_core.py:48-54`; `code/train_deploy_methods.py:48-51` |
| Recipe A (CPDINO ViT-B, whole frame) | Independent horizontal and vertical flips, p 0.5 each, drawn from (seed, epoch, row). Nothing else | `E/oof_recipeA_20261027/staging_regression_zero/phase2_backbone_ssl_regression_20261020.py:288-306` |
| DINOv3-B/16, CV-weighted DINOv3-B, ViT-L/16, attention-pool DINOv3-B | The same two flips. Every other switch is off in the launch files: `transpose_p` 0, `zoom` 1.0, `appear` 0, `shading` false, `drop_path` 0, no window, no teacher. `AUG_BUNDLE` is not set in the full-data launch lines | `E/lane_cd_20261108/arms/vitl16_f99_s46/launch.json`; `E/lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46/launch.json`; `E/lane_hex_20261108/source/gpu1_deploy_queue.sh:18` |
| SSL5 pre-training (two views per frame) | Independent flips, gain uniform 0.9-1.1, offset uniform +-0.03. No crop, resize or mask | `E/phase2_crossfold_20261015/source/phase2_lejepa_20261003.py:122-134` |
| Point detectors (`sg2`, `sg2pre`) | The four exact flips (identity, H, V, HV), with marker coordinates flipped accordingly. External set resampled by 1.296 (fixed, not random) | `green_sg2_specialist_20261028.py:3,158,195`; `green_external_srep_20261030.py:232-244` |
| Cell-graph networks | None found (dropout 0.1 only) | `graph_train_cv_20261108.py:14` |
| Cellpose-SAM segmentation | Not trained | |

No mixup, CutMix, colour jitter, gamma, noise, blur, elastic warp or random scale is in any adopted run.

### 2.2 Test-time augmentation in the package

| member | views | code |
|---|---|---|
| Phase-1 five | 5 tiles (4 corners + centre of the central 512) x 8 dihedral transforms = 40 views; mean in the standardised log space | `final_candidate_core.py:37-45,92-99` |
| Recipe A, CV/HEX and CD (tier p) | 4 flips at epoch 20, mean of log predictions, arithmetic mean over seeds 46/47 | `E/lane_cellroute_20261109/inference/members.py:67-77,97-99` |
| DINOv3-B for CD | 4 flips x epochs 16-20 x seeds 46/47 | `members.py:112-117` |
| DINOv3-B for CV | single view, epochs 16-20 | `members.py:120-122` |
| ViT-L/16 (tier p) | 4 flips at `final.pt`, seeds 46/47 | `members.py:125-128` |
| Attention-pool DINOv3-B, CV-weighted DINOv3-B | single view, epochs 8-12 | `members.py:162-180` |
| Pre-only detectors | mean heat map of 4 flips (flips undone), plus the single-view map | `source/deploy_hexkonan_image.py:37-49` |
| Other 13 detectors | single view | `deploy_hexkonan_image.py:81-85` |
| Cellpose-SAM segmentation | none (`augment` False by default) | `recipe_infer.py:56` |

### 2.3 Augmentation experiments and their measured results

Lower is better; deltas are MAPE points against the stated baseline.

| experiment | setting | result | status | source |
|---|---|---|---|---|
| Bundle on recipe A | rotation +-10 deg (reflected border), gamma log-uniform 0.8-1.25, Gaussian blur sigma 0.3-2.0 with p 0.5, Gaussian noise sigma 1-7 with p 0.5, on the raw frame before CLAHE | fold 0, seed 46: mean -0.042 (CI -0.104, +0.019); CD -0.08, CV -0.07, HEX +0.03; added to the stack -0.005 | closed (one fold, one seed) | `E/meta_ceiling_20261103/source/recipeA_invariance_20261104.py:20-31`; `docs/ASSUMPTION_AUDIT_20261106.md` belief 4 |
| Same bundle on DINOv3-B (CV lane) | `AUG_BUNDLE=1`, seeds 46 and 47, five folds | stack weight 0; inside the seed mean -0.030 vs -0.037 without it | closed | `E/lane_cv_20261108/source/family_cv_20261108.py:45-52`; `final_summary_cv_20261113.md:79` |
| Measured noise and optics transfer (recipe A) | donor transfer of focus, grey levels and noise | fold 0: member +0.004 (CI -0.043, +0.049); added +0.002 / -0.001 | closed | wiki `exp-gpu0-measured-augmentation` |
| Geometric (recipe A) | transpose p 0.5 and zoom up to x1.12 with CD x 1/s^2 | fold 0, epochs 8-9: +0.116 | closed | `final_summary_backbone_20261113.md:72` |
| C-Mixup (DINOv3-B) | | member, folds 0+4: 9.233 vs 9.2485 (-0.0155); stack weight 0 in all three (CD +0.004 to +0.011, CV +0.0073, HEX +0.0153) | closed | `final_summary_backbone_20261113.md:73` |
| Appearance augmentation (`--appear`) | gamma, contrast, blur or unsharp sharpening, noise | never run | open, untested | `bb_runner_20261108.py:167-207`; `final_summary_backbone_20261113.md:97` |
| Four-flip TTA, DINOv3-B, CD | epochs 16-20 | -0.008 CD over single view | kept | `final_summary_cd_20261113.md:97` |
| Flip-TTA DINOv3-B member, CV | | stack weight 0 | dropped | `final_summary_cv_20261113.md:79` |
| Cell graphs on flip-TTA detector peaks, CV | four members | 10.145 vs 10.151; removing both graph quartets costs +0.0118 strict | kept | `final_summary_cv_20261113.md:70` |
| SSL view set | only flips, gain, offset were ever used | not varied | | `docs/ASSUMPTION_AUDIT_20261106.md` belief 1 |

## 3. Preprocessing

### 3.1 Raw file to network input

| family | train time | inference time | same? |
|---|---|---|---|
| Recipe A, DINOv3-B, ViT-L, attention-pool, CV-weighted DINOv3-B, recipe A tokens for the frozen heads | `.mha` read once with SimpleITK into the uint8 memmap (972x1296, one channel). Per frame: CLAHE clip 2.0 with a 40x30 tile grid on the whole frame; scale by the frame's 1st and 99th percentiles, clip to 0-1; flips; expand to 3 identical channels; replicate-pad 16 px right and 20 px bottom to 992x1312; ImageNet mean/std for DINOv3-B and ViT-L, no further normalisation for CPDINO; bf16 autocast. No resize, no tiling, no crop | The same steps on the frame read from the `.mha` | yes, same constants (`phase2_foundation_train_20261002.py:127-129,144-149`; `family_bb_20261108.py:43-53,182-191`; `members.py:23-40`) |
| Phase-1 crop members | Central 512x512 of the frame (cached); 256x256 crop; CLAHE clip 2.0 with an 8x8 grid on the crop; 1-99 percentile scaling of the crop; 3 identical channels; no further normalisation | Central 512x512; the 5 fixed tiles; the same CLAHE and scaling per tile | same pixel operations; crop positions differ (random vs 5 fixed) (`final_candidate_core.py:13,21-34`; `code/train_oof_h200.py:167-173`) |
| Point detectors | Same as recipe A (CLAHE 2.0, 40x30; 1-99 %; pad 16/20; 3 channels; no ImageNet normalisation). External images: resize x1.296 bicubic first, then CLAHE with ~32 px tiles, 1-99 %, edge-pad to a multiple of 16 | Same as recipe A; each frame is run as a batch of 4 so the bf16 result equals the stored training peaks | yes (`E/green_track_20261028/deploy/hex6_correction.py:34-40`; `deploy_hexkonan_image.py:26-34,81-85`; `green_external_srep_20261030.py:69-75`) |
| Cellpose-SAM segmentation | CLAHE clip 3.0 with an 8x8 grid on the whole raw frame, then `CellposeModel.eval` with `flow_threshold` 0.8, `cellprob_threshold` -2.0 and library defaults | Same call | yes (`E/lane_hex_20261108/source/seg_run.py:14-20`; `recipe_infer.py:55-56`) |
| Cell graphs, hex6, cell-route features | Built from detector peaks and mask centroids; node appearance and image statistics are read from the raw frame | Same per-frame functions | yes by construction (`E/lane_hex_20261108/source/hex_test_features.py:1-17`) |

Cellpose's own normalisation (cellpose 4.2.1.1 defaults, checked with `inspect.signature`): `normalize=True`, percentiles (1.0, 99.0), `sharpen_radius` 0, `smooth_radius` 0, `tile_norm_blocksize` 0, `invert` False, `diameter` None, `augment` False.

### 3.2 Contrast, edge and line enhancement

- CLAHE is the only enhancement: clip 2.0 with 40x30 tiles (whole-frame models and detectors), clip 2.0 with 8x8 tiles (Phase-1 crops), clip 3.0 with 8x8 tiles (segmentation input).
- **No line or edge enhancement exists**: a search of the training, feature and inference code read for this report for Sobel, Laplacian, Hessian, Frangi, Sato, Meijering, top-hat, morphology, unsharp, denoise, flat-field and vignetting terms found none in the adopted path. A Laplacian variance is computed only as an image statistic (`recipe_infer.py:32-35`), not applied to the input.
- Flat-field (shading) correction exists as a switch (`bb_runner_20261108.py:210-216`) and was tested: the cross-source offset went from 0.0825 to 0.0959 (worse); closed, not in the package (`final_summary_backbone_20261113.md`, shading line).
- No denoising, no sharpening, no vignetting correction in any adopted run.

### 3.3 Label-side transforms

- Whole-frame models: target = log(label), standardised with the training rows' log mean and SD (stored in each checkpoint as `log_mean`, `log_std`). Loss = 0.5 x L1 on the standardised log + 2.5 x mean |exp(error) - 1| (`phase2_foundation_train_20261002.py:159-161`). Original labels, no smoothing. The CV-weighted member weights the CV column x3 (`CD_LOSS=cvw3`).
- Phase-1 members: target = log(label) standardised with fixed constants (mean 7.9199, -0.9825, -0.6211; SD 0.1897, 0.1881, 0.2749). Labels are first averaged over the training frames that share a `slide_id` ("slidesmooth"), so the two eyes of one new-format slide get the same target; old-format rows are unchanged because they are singletons (`code/train_oof_h200.py:143-152`; `prepare_final9000_candidate.py:48-52`). Loss: log L1 (metric35: 0.5 log L1 + 0.5 relative error / 0.2).
- Rows with a zero label are never trained on; no label is clipped or replaced.
- After the networks: Phase-1 mean x fixed scale (0.9970, 0.9880, 0.9892); recipe A x per-metric scalars; log-space blends; a log-linear stretch c x exp(g x (log q - mu) + mu) (`cd_combine_20261110.py:14-20`).

### 3.4 Differences between training and inference

1. **Input guard (inference only).** A frame that is not 972x1296 gets the constants CD 2770, CV 0.36, HEX 0.53; there is no resize path. A non-uint8 frame is min-max scaled to 0-255; a multi-channel frame is averaged. None of these conversions occurs in training data (`inference/docker/gc_entry.py:19,23-32,58-65`).
2. **Phase-1 members: package weights differ from the OOF weights in rows.** OOF comes from fold models trained on 7,161-7,163 rows of the 8,954. The packaged weights were trained on 8,998 rows, including the 46 rows of 1.2, with labels smoothed over all rows. Their calibration scale comes from the OOF models.
3. **Precision.** Training and the reference inference use bf16. Below compute capability 8.0 (T4) the container uses fp16 and one detector copy, and Cellpose runs without bf16; measured impact +0.0066 on the strict score (`final_summary_deploy_20261113.md` section 1).
4. **Detector encoder for the frozen-head detectors.** OOF features use the fold's recipe A encoder; the package uses the full-data encoder (`deploy_hexkonan_image.py:68-78`).
5. **Fold fits vs full-data fits (all whole-frame families).** Fold fits exclude 0-12 training rows linked to validation by pixel-identical frames; the full-data fits keep all 28 such rows.
6. The pixel pipeline itself (CLAHE constants, percentile scaling, padding, channel handling, normalisation) is identical between training and inference for every family.

## 4. Not established

- Whether `final_train_ids.csv` is the organisers' file unchanged, and where its `slide_id` column comes from: not found in the code or docs read.
- Why the brief says 8,953: no source found.
- Exact number of rows with a non-empty cell graph used to train the graph networks: not computed.
- SSL rows of the DINOv3-B and ViT-L full-data builds (8,952): taken from the CD summary, not re-derived from code.
- Training data of the `s8_*`, `s8multi_*` and frozen-head detectors was not read file by file; only `sg2` and `sg2pre` were.
- That `AUG_BUNDLE` was unset for the full-data DINOv3 builds is inferred from the launch scripts; `launch.json` does not record environment variables.
- The ledger `ledger_20261111.jsonl` was not read line by line; results in 2.3 come from the lane summaries, the audit and the wiki pages.

## 5. Snippets behind the counts

All run with `/workspace/venv/bin/python` on the pod, CPU only, read-only.

```python
# 9,000 files, 9,000 label rows, 46 unmatched (1.1, 1.2)
ff = np.load('datasets/clear_ec_fullframe_20260904/fullframe_meta.npz')            # ids (8954,), Y (8954, 3)
csv = pd.read_csv('datasets/clear_ec_official/final_train_ids.csv', dtype=str, keep_default_na=False)
len(csv), csv.ID.nunique()                    # 9000, 9000
np.isin(csv.ID.values, ff['ids']).sum()       # 8954  -> 46 rows not in the cache
# ls datasets/clear_ec_fullframe_20260904/train_mha | wc -l  -> 9000

# zero labels (1.3)
num = csv[['CD', 'CV', 'HEX']].astype(float); (num <= 0).sum()     # CD 0, CV 0, HEX 2
(ff['Y'] <= 0).sum(0)                                              # [0 0 2]

# OOF table, folds, donors (1.1, 1.3)
z = np.load('experiments/recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True)
len(z['indices']), (z['target'] > 0).sum(0), np.bincount(z['fold'])   # 8954, [8954 8954 8952], [1791 1791 1791 1791 1790]
# donors via eval_oof_20261031.donors(): 6768 groups, sizes {1: 4583, 2: 2184, 3: 1}; old rows 4514 = 4514 groups; donors in >1 fold: 0

# pixel duplicates (1.3)
fp = np.load('experiments/ssl_screen_20260911/preflight/pixel_fingerprints.npz')['sha256']
u, c = np.unique(fp, return_counts=True); len(u), (c > 1).sum()       # 8940, 14
# the 46 extra .mha files: SimpleITK read, sha256 of the pixel bytes; 0 matches with fp, 0 among themselves

# split files (1.4): train / validation sizes
# ssl_screen_20260911/preflight/fold0.npz 7151/1791, fold1.npz 7157/1791; overnight_target_20260912/preflight/fold2.npz 7153/1791,
# fold3.npz 7161/1791; target_fold4_20260912/preflight/fold4.npz 7150/1790; historical_train 7161/7161/7163/7161/7162
# followup_20260902/runs/fixed_base_f{f}_.../split_indices.npz train 7161/7161/7163/7161/7162
# fulldata_recipeA_20261028/fulldata_regression_split.npz train 8952, validation 2

# reduced subsets (1.5)
g = np.load('experiments/submission2_search_20261103/green_features.npz', allow_pickle=True)
good = ~g['seen_leaked']; good.sum()                                   # 8934
(g['f_active10_valid'] & good & np.isfinite(g['f_active10'])).sum()    # 8927
F = np.load('experiments/lane_cellroute_20261109/feats/cellroute_feats.npz')   # P2 columns, row order idx
X, m = deploy_hexkonan.design(Fr, base); ok = y[:, 2] > 0
m.sum(), (good & m & ok).sum()                                         # 8845, 8825
c = np.load('experiments/lane_cd_20261108/reads/freeze_cd_oof_20261111.npz')
have = np.isfinite(c['mask_cd']) & (c['mask_cd'] > 0) & (c['mask_n_int'] >= 50); have.sum()   # 7704 (535 NaN, 715 below 50)
```
