# September 14 method

## Inputs and architecture

Each grayscale uint8 MHA is processed at native resolution. A Cellpose-SAM
encoder supplies a 256-channel feature map. Spatial mean pooling and LayerNorm
feed three heads: Linear(256,128), GELU, Dropout(0.2), Linear(128,1).
The model regresses CD, CV and HEX; it does not compute them from segmentation masks.

## Native views

The first five models use five native 256 × 256 tiles from the central
512 × 512 field. The sixth uses five quality-ranked, spatially separated tiles
from a 4 × 6 full-image grid. Ranking is label-free contrast/sharpness/saturation,
not ROI supervision. Eight rotations/reflections per tile give 40 views per model.

Tiles use CLAHE (2.0, 8 × 8), 1st/99th percentile normalization, clipping to [0,1]
and grayscale replication into three channels.

## Six-member recipe

| Member | Seed | Training crops | Objective | Selected updates |
|---|---:|---:|---|---:|
| seed42 | 42 | 1 | Standardized-log L1 | 3,576 |
| seed77 | 77 | 1 | Standardized-log L1 | 2,831 |
| seed2026 | 2026 | 1 | Standardized-log L1 | 2,980 |
| metric35 | 42 | 1 | Mixed log/relative error | 4,470 |
| predbag2_35 | 42 | 2 | Standardized-log L1 | 3,874 |
| full_quality_seed77 | 77 | 2 | Mixed log/relative error | 5,215 |

The first five retain slide-averaged targets and their original weights.
The sixth uses custom masked-EMA SSL (9,000 images, 6,000 updates), then
original per-image targets (8,998 valid rows). No external corpus is used in
this additional SSL stage. See [SSL recipe and release evidence](SSL_RELEASE_20260914.md).

Supervised optimization uses AdamW, batch 48, encoder LR 1e-5, head/norm LR 1e-3,
weight decay 1e-4, BF16 and gradient clipping at 1. The first three use 200-update
warmup then constant LR; other recipes use warmup/cosine with a 0.05 floor.
[Historical training recipes](../review_submission_20260910/TRAINING.md) preserve
the first-five source/initialization differences.

## Inference aggregation

Average standardized-log predictions over 40 views **within each member**.
Decode each member using `exp(z * LS + LM)`, average the six decoded vectors
equally, then multiply by the unchanged calibration:

| CD | CV | HEX |
|---:|---:|---:|
| 0.9970446110788349 | 0.987972130802936 | 0.9892234872488541 |

FP16 inference loads models sequentially and batches eight views.
CD is cells/mm²; CV and HEX remain fractions.

See the [technical report](../output/pdf/Lamda_KIM_CLEAR_EC_Technical_Report.pdf)
for exact constants, supporting comparisons and validation limitations.

[Detailed historical record](archive/snapshot_20260915/docs/METHODOLOGY.md) — retained for provenance, not reviewer setup.
