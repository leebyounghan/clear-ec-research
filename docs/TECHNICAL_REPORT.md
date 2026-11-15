# CLEAR-EC Phase II: Technical Report (tier p3b)

## 1. Task and score

For each corneal endothelium specular microscopy image (972 x 1296, 8-bit) the method predicts cell density (CD, cells/mm²), the
coefficient of variation of cell area (CV) and hexagonality (HEX). The challenge score is the mean of the three per-metric MAPE values.

## 2. Approach in brief

Each metric has its own stack. Several deep regressors and cell-level pipelines produce candidate values; a fitted combination (log-space
blend for CD and CV, nested ridge regression for HEX) turns them into one prediction. All combination parameters were fitted out of fold
and then refitted on all training rows with the same procedure.

The building blocks are:

- **Whole-frame regressors** on ViT backbones (ViT-B/8, DINOv3 ViT-B/16, DINOv3 ViT-L/16), trained on the full frame with CLAHE and
  percentile scaling, with four-flip test-time augmentation for the CD members.
- **Crop regressors** on the Cellpose-SAM encoder (central 512 x 512 crop).
- **Zero-shot segmentation** with Cellpose-SAM, giving cell counts, areas and shape features.
- **Cell detectors** (point heatmaps), pretrained on the public Topcon specular set and fine-tuned on 20 frames with expert cell markers.
- **Graph networks** over detected cell centres for CV and HEX, and frozen token-statistic heads on the ViT-B/8 encoder for CV.

## 3. Stacks

### 3.1 CD

| Slot | Members | Weight |
|---|---|---:|
| P1 | five Cellpose-SAM crop regressors | 0.25 |
| A (recipe A) | two ViT-B/8 whole-frame regressors, seeds 46 and 47 | 0.40 |
| D | one DINOv3 ViT-B/16 regressor, checkpoints of epochs 16-20 | 0.05 |
| V | two DINOv3 ViT-L/16 regressors, seeds 46 and 47, geometric mean | 0.30 |

The blend is a weighted mean of log predictions. Then:

1. **Mask density.** Cellpose-SAM gives an interior cell count per area. It is mixed in with weight 0.1 when at least 50 interior cells
   are found.
2. **Stretch.** A log-affine map with slope 1.075, fitted to minimise MAPE.

The difference from the previous build (tier p) is the V slot: both ViT-L members are retrained on all labelled rows with an encoder
learning rate of 2e-5 instead of 1e-5 (12 epochs, self-supervised initialisation, EMA), and the second DINOv3 ViT-B/16 seed was dropped
from slot D to save run time.

### 3.2 CV

The base CV pipeline plus six log corrections: a DINOv3 ViT-B/16 term (0.125), three frozen token-statistic heads on the ViT-B/8
encoder (spectrum 0.25, covariance square root 0.375, autocorrelation 0.25), and two quartets of graph networks on detected cell
centres (test-time-augmented peaks 1.0, order-loss 0.25).

### 3.3 HEX

One nested ridge regression over 38 correction columns: cell detector outputs, a hexagonality corrector, graph networks, Cellpose-SAM
segmentation features and an attention-pooled DINOv3 ViT-B/16 column. A last step (cell route) corrects the value with cell-level
features.

## 4. Training data and validation

- 9,000 labelled training images; 8,954 are in the whole-frame training cache (46 were lost to an ID mismatch). Two rows with HEX 0 are
  excluded from CV and HEX.
- Five folds grouped by donor (6,768 donor groups; no group spans two folds).
- The 20 frames with expert cell markers train the detectors and are excluded from every stack fit (8,934 rows remain).
- Strict out-of-fold protocol: every fitted quantity (weights, scalars, stretch, model choices) is refitted without the scored fold.
  Stack comparisons also use a repeated grouped 10-fold fit with a donor-clustered bootstrap interval.

## 5. Results

Out-of-fold MAPE (%), strict protocol.

| Model | Mean | CD | CV | HEX |
|---|---:|---:|---:|---:|
| **p3b (this submission)** | **8.6596** | 6.1479 | 10.1115 | 9.7194 |
| tier p (previous build) | 8.6682 | 6.1737 | 10.1115 | 9.7194 |
| submission 1 | 8.877 | 6.312 | 10.204 | 10.115 |

Submission 1 scored 8.9675 on the hidden test set (CD 6.250, CV 10.143, HEX 10.510).

Evidence for the ViT-L change (CD):

| Check | Result |
|---|---|
| p3b against tier p | -0.0258 (95 % CI -0.040 to -0.012), lower in every fold; -0.0306 on the 10-fold fit |
| one seed put back to 1e-5 | +0.0139 (CI +0.005 to +0.023), worse in every fold |
| full-data members against their fold models (CD level) | seed 46 -0.29 %, seed 47 +0.57 % |

## 6. Code organisation

All code is under `src/experiments/`, at the paths it had in the research tree, because the fitted parameter files refer to these paths.

| Path | Role |
|---|---|
| `lane_cellroute_20261109/inference/docker/gc_entry.py` | container entry: reads the input, runs the recipe, writes the three outputs, falls back to constants on failure |
| `lane_cellroute_20261109/inference/recipe_infer.py` | `Recipe`: loads the manifest and parameters and runs the stages (segmentation, P1, detectors, CV parts, HEX features, hexagonality corrector) and the three stacks |
| `lane_cellroute_20261109/inference/members.py` | whole-frame regressors: preprocessing, model construction, flip test-time augmentation, checkpoint log-mean |
| `lane_cellroute_20261109/inference/pack_hooks.py` | loading hooks: fp16 checkpoints cast back to fp32, base weights not read, files shared by stages read once |
| `lane_cellroute_20261109/inference/manifests/deploy_p3b.json` | which checkpoints each stage reads |
| `lane_cellroute_20261109/inference/params/deploy_p3b.json` | fitted stack parameters of p3b |
| `lane_cd_20261108`, `lane_cv_20261108`, `lane_hex_20261108`, `green_track_20261028`, `meta_ceiling_20261103`, ... | per-metric modules and their deployed parameter files, imported by the recipe |
| `phase2_foundation_20261002/runtime` | vendored timm 1.0.20 and DINOv3 backbone code (first on `PYTHONPATH`) |

Files of other tiers (`deploy_p.json`, `deploy_t1.json`, ...) remain in the folders; the image copies the p3b files to `deploy.json`
at build time.

## 7. Implementation details

- **One image per container.** Each stage loads its models, runs, and frees GPU memory before the next stage (peak 4.3 GB instead of
  about 17 GB). Weights shared by several stages are read once per image.
- **Precision.** On a T4 the code uses fp16 autocast and one detector copy; on newer GPUs bf16. The T4 path changed out-of-fold CD by
  +0.0066 on a 200-row check.
- **fp16 storage.** Regression checkpoints are stored in fp16 and cast back at load time (median output change at most 0.009 %).
- **Runs as uid 10001 without network.** The build checks that every code file is readable by that user.
- **Fallback.** Any failure on an image writes CD 2770.0, CV 0.36, HEX 0.53 for that image so that the evaluation continues.

## 8. Limitations

- The run time on the challenge platform was not measured by us. The T4 estimate (258-296 s per image) is close to the 300 s limit.
- Eleven training rows have a CD error above 100 % and add 0.263 to the CD MAPE; several look like label entry errors. They were kept.
- The gap between out-of-fold and test score seen for submission 1 (HEX +0.395) may repeat.
