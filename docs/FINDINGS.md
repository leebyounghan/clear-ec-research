# Findings — what worked, what failed, and why

> **2026-09-02 qualification:** The historical "proven noise floor" claims below
> are not established. Shared crop preprocessing, a small ROI reference set, and
> a newly discovered optimizer/gradient-clearing defect are alternative causes
> of the observed plateau. See [current validation plan](FOLLOWUP_20260902.md)
> before treating historical rejections as definitive.

This document records historical experiments, not universal limits on the task.
Read [the September 5 correction and validation run](DEPLOY_REVIEW_20260905.md)
before selecting a new experiment or submission candidate.

## Historical observation: a CV/HEX plateau, not a proven noise floor

CV (~10.4%) and HEX (~10.0%) MAPE changed little in the early experiments.
The following observations motivated a label-noise hypothesis, but do not isolate
label noise from shared preprocessing, region mismatch, or optimization:

1. **Model-invariant errors** (`code/eval_dino_ensemble.py`). A DINOv2-large regression member — a
   *completely different pretraining prior* (natural-image SSL vs cpsam's cell segmentation) — makes
   **near-identical per-image errors**. Error correlation vs the s2 member: **CD 0.939, CV 0.992, HEX 0.982**.
   Adding it to the ensemble did not help (historical test 8.92 → 8.98).
   Shared cropping can also produce correlated errors; this does not reject all backbone changes.

2. **ROI-invariant errors** (`code/roi_ceiling_test.py`). Feeding the model the annotator's **true ROI
   location** (vs central-512) on the 25 green images improved CV by only **0.35** and HEX by **0.22**.
   This small, historically reused sample does not establish the effect across the full dataset.

**Interpretation.** Region sampling and label variability may limit performance;
the diagnostic window spread in `results/roi_region_diag.json` is not itself a
lower bound on achievable MAPE. The September 5 fixed five-model ensemble improved
all three metrics in ordinary OOF, including CV 10.434774 → 10.382686 and HEX
10.226306 → 10.185301. These are small internal gains, not leaderboard predictions.
Do not restrict future work to CD or reject full-resolution region-aware methods
solely because of this historical plateau.

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
