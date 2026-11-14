# Freeze candidate: cell-route HEX step (Konan statistics of a union point set), 2026-11-09

## What it is and what it gives
- A nested ridge on the log HEX residual of the HEX lane's final HEX.
- Inputs are Konan k=0 / k=2 cell statistics (CV, 6-sided share, cell count) of a union point set:
  - three pre-only sg2 detectors (seeds 46, 47, 48) with 4-flip TTA;
  - Cellpose-SAM mask centroids;
  - cross-source vote within 0.45 spacing, >= 1 source.
- The statistics are read in the 3 legible 380x300 windows and over the full frame.
- **Five-fold OOF on the current HEX chain (11-10, `lane_hex_20261108/preds/chain_mtcnn5fix.npz`, 9.7358): HEX 9.7283 (-0.0075).** Folds: -0.017 / -0.010 / -0.009 / +0.002 / -0.003; folds 0+4 -0.010. OOF file: `out/corr_hex_on_mtcnn5fix.npz`.
- On the previous chain (`chain_cvw3`, 9.7380), the same recipe gave 9.7315 (-0.0065), folds -0.020 / -0.008 / -0.008 / +0.007 / -0.004.
- Deploy parameters are refit on the new base: `out/deploy_hexkonan_params.json`. The cvw3-base parameters are archived as `out/archive_deploy_hexkonan_params_chain_cvw3_20261110.json`.
- The point set was pre-registered from the 25-frame gate. Alpha (100) and the 50-cell minimum were fixed, never tuned.
- Caveat: small and not robust across versions. Other versions built on similar features read between -0.0015 and +0.010 (`RESULTS.tsv`).

## Inference path (per test image)
1. Inputs it shares with the HEX lane:
   - the raw frame;
   - the HEX lane's final HEX;
   - the HEX lane's own loose Cellpose-SAM mask (cpsam_v2, flow 0.8, cellprob -2, CLAHE clip 3, 8x8 tiles). No second segmentation.
2. `source/deploy_hexkonan_image.py`, `HexKonan([sg2pre_s46, sg2pre_s47, sg2pre_s48], out/deploy_hexkonan_params.json)`:
   - preprocessing = the deployed hex6 `preprocess`;
   - per detector, the mean heatmap over 4 exact flips; peaks with `SG.peaks2` (>= 0.2, NMS 12 px), per detector and for the mean of the 3 heatmaps.
3. `source/deploy_hexkonan.py`:
   - `seg_points(mask)` gives the mask centroids;
   - `features()` gives 10 features;
   - `correct()` applies the ridge with all-fold parameters (`out/deploy_hexkonan_params.json`, 8825 training rows).
   - Images with < 50 full-frame cells or non-finite features keep the base HEX.
4. Code to copy: `source/deploy_hexkonan.py`, `source/deploy_hexkonan_image.py`, `source/cr_lib.py`, plus the green-track `deploy/` and `source/` modules the hex6 corrector already uses.

## Checks done (CPU)

| Check | Result |
|---|---|
| `features()` from stored peaks and centroids vs the OOF feature file (300 random frames) | max abs diff 0, NaN pattern identical |
| Per-fold ridge via `correct()` vs the stored OOF predictions `out/corr_hex_a100.npz` | max rel diff 7e-16 |
| `seg_points()` vs the stored centroids (2 frames) | identical |
| Image path, fold-0 detectors on CPU fp32, 2 frames, vs the stored GPU bf16 TTA peaks | 97-99 % of peaks within 2 px, mean offset ~0.1 px; features within 1-3 % (threshold-edge numerics) |

## Must be trained on full data (measured costs)

| Item | How | Measured cost |
|---|---|---|
| `sg2pre_s46` | exists: `green_track_20261028/deploy/detectors/sg2pre_s46.pt` (full-data recipe A s46 encoder) | 0 |
| `sg2pre_s47`, `sg2pre_s48` | **Use the green-track pair `green_track_20261028/deploy/detectors/sg2pre_s47.pt`, `sg2pre_s48.pt` (10-30), the same pair the CV lane names.** Decided 11-10: the rehearsal found two pairs. The cell-route duplicate trained 11-09 (same recipe, init and seed; different weights from GPU nondeterminism) was moved to `deploy/archive_duplicate_pair_20261110/`, not deleted. On 4 CPU frames the green-track pair's features track the OOF features at least as well as the duplicate (median per-frame max rel diff 0.11 vs 0.52; both dominated by window flips). Command used for both pairs: `green_external_srep_20261030.py pretrain --kind sg2 --seed 47 / 48 --updates 2000 --scale 1.296 --encoder experiments/fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt --deploy-out experiments/lane_cellroute_20261109/deploy/detectors/sg2pre_s4X.pt` | 276-280 s each on one H200 (s46 deploy 279.7 s; s47 fold 0 275.9 s), about 10 GPU-min in total |
| Ridge parameters | `source/fit_deploy_hexkonan.py` on the stored OOF features; rerun if the HEX lane's deployed chain changes after `chain_cvw3` | CPU seconds |

Measured compute per image:
- 12 sg2 forward passes, about 1 s on an H200 (the hex6 corrector measured 0.80-0.88 s for 10 passes);
- union and Konan geometry, about 0.5 s on CPU;
- the shared Cellpose-SAM pass, about 1.0 s GPU (HEX lane measurement).

## Full-data detectors vs OOF (200 random frames, GPU 1, `source/check200.py`, `out/check200.json`; run 11-09 with the now-archived duplicate pair)
- Features from sg2pre_s46/s47/s48 (full data) against the OOF features from the fold detectors:
  - window features: r 0.88-0.96, median relative difference 2.8-7.9 %;
  - full-frame features: r 0.95-0.99, median relative difference 1.5-4.5 %.
- Corrected HEX with the deploy parameters is within a median 0.15 % (|log| 0.0015) of the OOF correction. The correction itself moves HEX by a median of about 0.5 %.
- The end-to-end image path ran 2.1 s per frame on GPU 1, sharing it with the HEX lane's CPSAM run.

## Open before the rehearsal
- The final HEX chain: `chain_cvw3` now. If the five-fold CPSAM member is kept after the HEX lane's 11-10 11:00 EST read, the base becomes `preds/chain_mtcnn5fix.npz`; then rerun `source/fit_deploy_hexkonan.py` with that base (CPU seconds) and re-read the five folds.
