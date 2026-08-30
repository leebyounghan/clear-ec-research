# Findings — what worked, what failed, and why

This is the most important doc for anyone continuing the work: **it records the dead ends so you don't
repeat them**, and the one proven structural fact about the task.

## ⭐ The central finding: CV/HEX are at a proven noise floor

CV (~10.4%) and HEX (~10.0%) MAPE have not moved across dozens of variations. Two independent experiments
prove this is **the data/label, not the model**:

1. **Model-invariant errors** (`code/eval_dino_ensemble.py`). A DINOv2-large regression member — a
   *completely different pretraining prior* (natural-image SSL vs cpsam's cell segmentation) — makes
   **near-identical per-image errors**. Error correlation vs the s2 member: **CD 0.939, CV 0.992, HEX 0.982**.
   Adding it to the ensemble did not help (test 8.92 → 8.98). Two unrelated backbones hitting the same wall
   from the same side ⇒ the residual is not model-specific.

2. **ROI-invariant errors** (`code/roi_ceiling_test.py`). Feeding the model the annotator's **true ROI
   location** (vs central-512) on the 25 green images improved CV by only **0.35** and HEX by **0.22**.
   Knowing the exact scored region barely helps.

**Why.** CV = std/mean over ~125 hand-picked cell areas; the GT is a single noisy draw from a distribution
whose intra-image window spread is ~32% (`code/roi_region_diag.py`, `results/roi_region_diag.json`). A
regressor predicts the *expected* CV; MAPE against a single noisy draw has an irreducible floor ≈ the
sampling relative-std. This matches the literature (human inter-reader ICC ≤ 0.27; 125-cell sampling gives
CV ±10–20%). **Consequence: stop trying to beat CV/HEX with better/more-diverse models or ROI localization.
CD is the only metric with real headroom** (its error correlation 0.939 < 0.99 ⇒ it has model-specific
variance to reduce).

## Adopted (improvements that survived held-out A/B)

| Idea | Effect | Where |
|---|---|---|
| **Per-metric greedy ensemble** (CD s2+at / CV s2 / HEX at+xHEX+cl) | test 8.92 | `ens_specialist.py` |
| **Slide-averaged targets** (label denoising) | **−0.20 test OV** vs raw | `train_slidesmooth.py`, `Y_slidesmooth*.npy` |
| **β=−1 calibration** on CV/HEX (×0.991 / ×0.993) | **−0.03** | `submission/inference.py` `CALIB` |
| **40-view C5×8 TTA** | standard, kept | all eval/inference |
| **Log-space targets + log-L1 loss** | robust to skew | `train_finetune.py` |
| **All-9000 retrain** for the final (no holdout) | more data; unmeasurable internally | `train_final9000.py` |

## Rejected (with the number, so nobody retries blindly)

| Idea | Result | Why it failed | Script |
|---|---|---|---|
| **Self-training / pseudo-mask scaling** | region-eval 17.8→18.6, CV diverged 21→26 | pseudo-masks smooth cell areas → CV underestimation entrenched | `finetune_cpsam_pseudo.py`, `round_*.py` |
| **Full-frame regression** (train on whole 972×1296, grid-agg) | test 9.18 | outer frame adds noise, not region signal | `train_fullframe.py` |
| **DINOv2 decorrelated backbone** | test 8.98 (worse); err corr 0.99 | not actually decorrelated — see central finding | `train_dino.py`, `eval_dino_ensemble.py` |
| **Learned/heuristic ROI recovery** | median IoU 0.20 vs true box | many windows accidentally match GT; ROI unrecoverable from metrics | `green_roi_recovery.py` |
| **ROI localization ceiling** | true ROI helps CV only 0.35 | CV error is sampling noise, not wrong-region | `roi_ceiling_test.py` |
| **fNLA edge-map scalar fusion** | blend weight → 1.0 (fNLA ignored) | fNLA seg CV 14–18% > regression 10.4% on this data | `fusion_metalearn.py`, `fnla_pool_scalars.py` |
| **FFT/spectral CD** (Selig/Foracchia) | CD MAPE 11–17%; err corr 0.66 | decorrelated but too weak; blend weight → 0 | `fft_morphometry*.py`, `stack_fft_test.py` |
| **Point-detector CD (hybrid)** | 16.6% best | worse than regression | `train_pointdet*.py`, `eval_point_*.py` |
| **Watershed CV** | 207% | segmentation quality on this data | `eval_watershed_cv.py` |
| **Per-metric specialists** | only xHEX helped marginally; spec_cd/cv no-win | multitask heads already near-optimal | `train_specialist.py` |
| **Marker-seeded watershed** (perfect dots) | CV still 11–17% | edge evidence missing in blurry regions even with dots | `green_marker_seeded.py` |

## Data facts worth knowing

- **Slide structure** (`final_train_ids.csv`, `slide_id`): 6771 slides, 2228 with ≥2 frames (max 3).
  Intra-slide GT rel-std: CD 2.5% / CV 5.3% / HEX 5.5%. This is why slide-averaging helps — but it can't be
  exploited at inference (the container gets one image, no slide grouping).
- **Green reference set** (25 images, `datasets/.../green_eval/`): the only images with a known ROI box +
  cell markers. Use for ROI/segmentation ceiling tests, not for training (too few).
- **fNLA** (domain-specific edge segmenter, Keras/TF): edge maps cached in `pool_edge_u8.dat`. On-domain
  paper SOTA is CV/HEX 5.7% (Vigueras TVST 2020), but on *this* dataset it underperforms regression because
  image quality varies and the ROI is hidden.
