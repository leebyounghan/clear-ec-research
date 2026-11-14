# CD freeze candidate (CD lane, GPU 0), written 2026-11-09

Five-fold nested OOF **CD 6.1741** against the best pipeline's **6.2639** (best_pipeline_20261106 `REF_BEST`, CD column): **-0.090**, lower in every fold (-0.099, -0.064, -0.053, -0.122, -0.111). Only the CD column changes; CV and HEX are untouched.

Version 3 (2026-11-11, coordinator seed audit): the DINOv3-B member is the geometric mean of seeds 46 and 47 (no seed picked). Version 2 used DINOv3 s46 alone: 6.1726 (+0.0016 for the seed mean: s46 was a slightly lucky seed); its files `reads/freeze_cd_*_20261109.*` stay as they were. Version 2 (11-09) made ViT-L the s46/s47 mean; version 1 (ViT-L s46 alone, 6.1775) is in `archive/freeze_v1_vitl_s46/`.

Reproduce (GPU or CPU pod, repo root, CPU, seconds):

```
/workspace/venv/bin/python experiments/lane_cd_20261108/source/cd_combine_20261110.py check experiments/lane_cd_20261108/reads/freeze_cd_oof_20261111.npz experiments/lane_cd_20261108/reads/freeze_cd_params_20261111.json
-> {"cd_oof_recomputed": 6.1741, "cd_oof_stored": 6.1741, "max_rel_diff_vs_stored": 0.0, "baseline_best_pipeline": 6.2639}
```

## Files (pod, under `experiments/lane_cd_20261108/`)

| File | What |
|---|---|
| `reads/freeze_cd_oof_20261111.npz` | Per row (8954, hybrid_oof order): `idx`, `ids`, `fold`, `y_cd`, member inputs `P1`, `A`, `D`, `V`, `mask_cd`, `mask_n_int`, final `oof_cd`, reference `ref_cd` |
| `reads/freeze_cd_params_20261111.json` | `per_fold` params (each fit on the other four folds; used by `check`), `deploy` params (same search on all five folds), recipe A CD scalars |
| `source/cd_combine_20261110.py` | `combine(P1, A, D, V, mask_cd, mask_n_int, params)`: numpy only; `check` recomputes the OOF value |
| `source/cd_freeze_20261111.py` | Builds the two files above (search grids, nesting); copy of `cd_freeze_20261109.py` with D = seed mean |

## Combine rule and deploy parameters

`log q = 0.25 log P1 + 0.45 log A + 0.10 log D + 0.20 log V`; if `mask_n_int >= 50`: `log q += 0.10 * (2.0602 + 0.74573 * log mask_cd - log q)`; then `CD = 0.99704 * exp(1.075 * (log q - 7.91822) + 7.91822)` (exact constants in the params JSON `deploy`).

Per-fold weights (P1, A, D, V) are 0.25/0.40-0.45/0.10-0.15/0.20-0.25; the mask weight is 0.1 in every fold. Exact values in the params JSON.

## Members

| Key | Member | Seeds | OOF predictions | Deploy model | Full-data training | Time |
|---|---|---|---|---|---|---|
| P1 | Hybrid B's Phase-1 half: crop CPSAM members, calibrated (x SCALE CD 0.99704) | five: seed42, seed77, seed2026, metric35, predbag2_35 | `meta_ceiling_20261103/source/ceiling_meta_cpu_20261103.py` `HIST` paths (`best_oof.npz` per fold); combined value is `hybrid_oof.npz` `historical_five_calibrated` | Phase-1 app `submission1_test_20261030/source/phase1_app/base_inference.py`, `predict(image, model, "candidate5.json")` = the same five full-data members, mean x calibration | none (existing package) | 0 |
| A | Recipe A (CPDINO ViT-B s8, SSL5, std pool), per seed `exp(0.5 log TTA4@e20 + 0.5 mean log single@e8..19)`, seeds averaged, times the recipe A CD scalar | s46, s47 | `C.RECIPE[46|47][fold]/runs/fold*/cpdino_vitb/` `predictions_tta4.npz` and `predictions_epoch_08..19.npz` | `fulldata_recipeA_20261028/regression_std_s46|s47/runs/fold99/cpdino_vitb/epoch_08.pt ... epoch_20.pt`; scalar at deploy 0.99139 (mean of fold scalars; exact value in the params JSON) | done (2026-10-28) | 0 |
| D | DINOv3-B/16 stride 16 (family runner, SSL5 + 20 epochs, schedule10), four-flip TTA over epochs 16-20 per seed, then the geometric mean of the two seeds | s46, s47 (s47 reuses each fold's s46 SSL) | f0/f4 `lane_gpu3_20261106/runs/dinov3_b16_f{0,4}_s46`, f1-3 `lane_cd_20261108/arms/dinov3_b16_f{1,2,3}_s46`; s47: f0/f4 `lane_gpu2_20261106/runs/dinov3_b16_f{0,4}_s47`, f1-3 `lane_cd_20261108/arms/dinov3_b16_f{1,2,3}_s47`; TTA in `lane_cd_20261108/tta/dinov3_s4{6,7}_f{f}_tta4_e16_20.npz` | s46 **built** `arms/dinov3_b16_f99_s46b/runs/fold99/dinov3_b16/epoch_16..20.pt` (SSL in `arms/dinov3_b16_f99_s46/ssl/last.pt`); s47 **built** `arms/dinov3_b16_f99_s47/runs/fold99/dinov3_b16/epoch_16..20.pt` (11-11 05:26 EST, 1.1 h, SSL reused; sha256 and held-image D reference values in `reads/verify_f99_dinov3_20261111.json`) | SSL: `family_cd_20261108.py all --family dinov3_b16 --fold 99 --seed 46 --root arms/dinov3_b16_f99_s46 --workers 4` (its regression died on the NaN bug, fixed); regression: `family_cd_20261108.py all --family dinov3_b16 --fold 99 --seed 46 --ssl-from arms/dinov3_b16_f99_s46/ssl/last.pt --root arms/dinov3_b16_f99_s46b --workers 4` | **measured 1.9 h** (SSL 2703 s + regression 4071 s), 11-10 |
| V | ViT-L/16 DINOv3 (timm `vit_large_patch16_dinov3.lvd1689m`), SSL5 + 12 epochs, raw track, per seed `exp(0.5 mean log single@e8..11 + 0.5 log TTA4@e12)`, then the geometric mean of the two seeds | s46, s47 (s47 reuses each fold's s46 SSL) | s46: f0/f4 `lane_backbone_20261108/runs/vitl16_avg_f{0,4}_s46/raw/runs/fold*/dinov3_l16t`, f1-3 `lane_cd_20261108/arms/vitl16_f{1,2,3}_s46/raw/...`; s47: `lane_cd_20261108/arms/vitl16_f{0..4}_s47/raw/...` | s46 **built** `arms/vitl16_f99_s46/raw/runs/fold99/dinov3_l16t/` (11-10 18:54 EST, sha256 final.pt fbb0eca8...); s47 **built** `arms/vitl16_f99_s47/raw/runs/fold99/dinov3_l16t/` (11-10 21:37 EST, GPU 3, sha256 final.pt d43c3a73...); each `epoch_08..11.pt` + `final.pt` (= epoch 12; no epoch_12.pt is written) | queue lines `vitl16_f99_s46` (`family_bb_cd_20261108.py ssl --family dinov3_l16t --fold 99 --seed 46 ...` then `bb_runner_cd_20261108.py --family dinov3_l16t --fold 99 --seed 46 --epochs 12 --micro 8 --ema 0.998,0.999 --swa-start 6 --tta-final --keep-epochs 8,9,10,11,12`) and `vitl16_f99_s47` (bb_runner only, `--seed 47 --ssl arms/vitl16_f99_s46/ssl/last.pt`) | s46 **measured 4.0 h** (SSL 7475 s + regression ~6800 s, GPU 0 alone); s47 regression only, ~2 h alone (on a shared GPU 3 it runs ~28 min/epoch, ~6 h) |
| mask | Cellpose-SAM `cpsam_models/cpsam_v2`, zero-shot, flow 0.8, cellprob -2, CLAHE; `seg_run.feats()` -> `cd_mm2`, `n_int` | - | `lane_hex_20261108/seg_loose/c*.npz` (`F` columns `cd_mm2`, `n_int`; `frames` = cache rows) | same model, no training | none | 0 |

Fold 99 = `fulldata_recipeA_20261028/fulldata_regression_split.npz` (train 8952, the two HEX-zero rows held), the same split as the recipe A full-data members.

## Inference entry point and inputs

Input: the image only (1296 x 972). No detector peaks are used by the CD recipe.

1. P1 and A: code in `submission1_test_20261030/source/` (`hybrid_inference_20261030.py`, `phase1_app/`). Do not use `predict_hybrid` as shipped (six Phase-1 members, recipe A s46/47/48 with three-seed scalars); see "Deployment must use" below.
2. D: the family runner's evaluate path with the four flips, once per kept epoch 16-20 (`source/family_tta_20261108.py` shows the exact forward; preprocessing is the recipe A one: CLAHE 2.0 tiles 32, percentile 1-99, replicate pad 16/20, ImageNet normalisation).
3. V: `bb_runner_cd_20261108.py` evaluate path: single view at epochs 8-11, `evaluate(tta=True)` with `final.pt` (epoch 12).
4. Mask: `lane_hex_20261108/source/seg_run.py` model call with the settings above and `feats(M)`; use `cd_mm2` only when `n_int >= 50`.
5. `combine()` with `deploy` params.

## Deployment must use (to match the OOF recipe; 11-10, after the integration rehearsal)

| Input | Use exactly | Not this |
|---|---|---|
| P1 | Phase-1 `base_inference.predict(image, model, "candidate5.json")`: members seed42.pt, seed77.pt, seed2026.pt, metric35.pt, predbag2_35.pt (full-data; files in `deploy_review_20260905/staged/model`, sha256 equal to submission 1's members, e.g. predbag2_35.pt c3988da6...), mean x calibration **[0.9970446110788349, 0.987972130802936, 0.9892234872488541]** (the OOF `SCALE`, `recipeA_analysis_20261028/source/hybrid_oof_20261028.py:19,109`); CD column. The staged `candidate5.json` carries [0.99743, 0.98675, 0.98887]: replace it with SCALE (CD +0.039 % on P1 otherwise) | `inference.py` / `candidate6_*.json` (adds `full_quality_seed77.pt`, six members); `deploy_methods_20260905/staged/model` (older predbag2_35.pt); the staged candidate5.json calibration as is |
| A | Recipe A full-data s46 and s47 only. Per seed: `exp(0.5 log TTA4(epoch_20.pt) + 0.5 mean_e log single_view(epoch_e.pt), e = 8..19)`; A = 0.5 (s46 + s47) x 0.99138607 (CD scalar, params JSON `recipeA_cd_scalar_deploy`) | s48; TTA at epoch 20 alone (the CV and HEX lanes' A); the shipped three-seed scalars (CD 0.9908) |
| D | DINOv3-B s46 and s47 full data: per seed four-flip TTA at each of epoch_16..20.pt, mean log over the 20 forwards; D = exp(0.5 (that of s46 + that of s47)). If s47 full data is not built by 11-11 15:00 EST, s46 alone (coordinator rule) | single view; picking one seed |
| V | ViT-L s46 and s47 full data, raw track: per seed `exp(0.5 mean_e log single_view(epoch_e.pt), e = 8..11 + 0.5 log TTA4(final.pt))`, then the geometric mean of the two seeds | "log mean of epochs 8-12" (the backbone rebuild sheet's read); EMA/SWA tracks |
| mask | Cellpose-SAM cpsam_v2, flow 0.8, cellprob -2, CLAHE; `feats()` `cd_mm2` used only when `n_int >= 50` | other segmentation settings |
| combine | `cd_combine_20261110.py combine()` with `reads/freeze_cd_params_20261111.json` `deploy` | the 11-09 params (DINOv3 s46 alone); per-fold params (OOF only) |

Full-data split for every CD member: fold 99 (`fulldata_recipeA_20261028/fulldata_regression_split.npz`, the same 8952 training rows as the backbone sheet's `--fold -1`).

## Known gaps

- The epoch 8-19 part of A means 12 extra single-view forwards per seed; there is no inference code for it yet (checkpoints exist). An SWA average of those epochs was not validated.
- D full-data model built (1.9 h). V s46 built (4.0 h, GPU 0, 11-10 18:54 EST); V s47 built (GPU 3, backbone lane; first launch OOM at 17:01, relaunched 17:26, done 21:37 EST, ~4.2 h on a shared GPU). D s47 built 11-11 05:26 EST (1.1 h). All CD members now have full-data models, each verified (sha256, strict load, held rows reproduced: `reads/verify_f99_vitl_20261111.json`, `reads/verify_f99_dinov3_20261111.json`); no CD build is pending.
