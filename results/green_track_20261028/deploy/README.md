# Submission-2 hex6 correction — deployment notes

Status (2026-10-31): ACTIVE configuration = nine fold-99 detectors + one pre-trained-only detector (parent
decision rule applied to the confirmation run; ACTIVE.json). Effect estimate: five-fold OOF LOFO on the hybrid
base B (0.5 recipe A two-seed + 0.5 Phase-1 five), 8925 non-leaked rows: HEX -0.093 pp, mean score
-0.031 pp, HEX improving on 5/5 folds, 3/5 folds <= -0.03 (per-fold mean deltas -0.047 / -0.031 / -0.042 /
-0.027 / -0.009). CD and CV are untouched. Not validated on the hidden test set. The previous nine-only
configuration (-0.027 pp) is kept as hex6_blend_params_frozen9.json.

## What it does
For one specular frame it runs ten point detectors (fine-tuned CellposeDINO ViT-B encoders + small heads),
finds cell-centre peaks, builds the Voronoi tessellation, computes the six-neighbour fraction of bounded cells
(hex6) per detector, averages log hex6, maps it to log HEX with the fitted a, b, and blends with the base HEX:
HEX_corr = exp((1 - w) log HEX_base + w (a + b mean_log_hex6)). Fallback: if no detector yields a value
(fewer than 20 peaks or fewer than 12 bounded cells for all ten), the base prediction is returned unchanged.

## Inputs / outputs
- `Hex6Corrector(deploy_dir)` loads once: the detectors named in `hex6_blend_params.json` (ten) and the parameters.
- `corrector(image_u8, base_pred) -> (corrected_pred, info)`: `image_u8` is the raw 972x1296 uint8 frame
  (the same array the regressor reads; preprocessing — CLAHE 2.0 with a (40, 30) tile grid and 1-99 percentile
  scaling — is done inside); `base_pred` is the final hybrid prediction `[CD, CV, HEX]` after its own
  calibration. `corrected_pred` differs only in HEX. `info` carries the feature and the per-detector hex6.
- Plug-in point: after the hybrid produces its calibrated HEX for the image, before writing the output JSON.

## Model files
| file | content | size |
|---|---|---|
| detectors/s8_s46.pt, s8_s47.pt, s8_s48.pt | stride-8 linear point heads + fine-tuned encoders (seeds 46-48) | ~340 MB each |
| detectors/sg2_s46.pt ... sg2_s50.pt | stride-2 pixel-shuffle heads + encoders (seeds 46-50) | ~340 MB each |
| detectors/s8multi_s47.pt | linear head on blocks 5/8/11 tokens + encoder | ~340 MB |
| detectors/sg2pre_s46.pt | stride-2 head pre-trained on the external srep point corpus only (no CLEAR-EC fine-tune) + encoder | ~340 MB |
| hex6_blend_params.json | ACTIVE: a = -0.2084, b = 0.5989, w = 0.28, ten families, fallback rule, LOFO numbers | 2 KB |
| hex6_blend_params_frozen9.json | previous nine-only parameters (a -0.2233, b 0.5835, w 0.26) | 2 KB |
Total ~3.4 GB. Each detector carries its full encoder, so the base recipe A checkpoint is not needed at
inference. The nine were trained from the full-data recipe A encoder
(experiments/fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt) on the 20
train-green images with the fold-clean recipe (1000 updates, threshold 0.35); sg2pre_s46 was pre-trained from
the same encoder on 307 srep images (Daniel et al. 2019; 2000 updates, hull-masked targets, 1.30x rescale) and
never fine-tuned on CLEAR-EC images, which is what makes its errors independent of the other nine.

## Runtime
0.80-0.88 s per image for ten detectors on one H200 (bf16, SDPA; unit test 0.80, regression test 0.88),
dominated by ten ViT-B forward passes at 992x1312; Voronoi and peak finding are milliseconds on CPU. Load
time ~15 s. Memory: ten encoders resident (~3.4 GB weights).

## Dependencies
torch, numpy, scipy (Delaunay/Voronoi), opencv-python (CLAHE), cellpose (CPDINO ViT-B definition) — the same
environment as the regressor. Code: `hex6_correction.py` plus the frozen modules in `../source/`
(green_point_features_20261027.py, green_sg2_specialist_20261028.py, green_s8_specialist_seed_20261028.py,
phase2_foundation_preflight_20261002.py, and their imports). Copy `deploy/` and `source/` together.

## Tests
- `test_hex6_correction.py`: structural checks on the 25 annotated images (needs the training raw memmap).
- `regression_test_25.py --fixture regression_fixture_25_v2.npz --out regression_result_v2.json` (fixture 19 MB,
  ACTIVE configuration): container-runnable; reproduces the activated run's corrected HEX on the 25 images
  within 0.5% relative and the feature within 0.02; exit code 0 on pass (2026-10-31: passed, zero drift).
  `regression_fixture_25.npz` is the nine-only fixture (use with hex6_blend_params_frozen9.json).

## Citations
See CITATIONS.md: Daniel et al., Sci Rep 2019 (srep_topcon_sp3000) is REQUIRED — the active configuration
includes a detector pre-trained on that corpus.
