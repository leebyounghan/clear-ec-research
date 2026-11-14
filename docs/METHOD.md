# Method: tier p3b

## Data

- 9,000 training images (972 x 1296, 8-bit, `.mha`) with CD, CV and HEX labels. 8,954 are in the whole-frame cache; 46 were lost to
  an ID string mismatch. Two rows with HEX equal to zero are excluded from CV and HEX.
- Five folds grouped by donor: 6,768 donor groups, none spanning two folds.
- 20 frames with expert cell markers train the cell detectors and are kept out of every stack fit, which leaves 8,934 rows.
- External data: the public Topcon specular set (385 images with cell points; 307 used to pretrain detectors).
- Preprocessing: CLAHE and 1-99 percentile scaling. Training augmentation of the whole-frame models: horizontal and vertical flips.

## Components

Each metric is predicted by a stack of members. Blends are done in log space.

### CD

| Slot | Members | Weight |
|---|---|---:|
| P1 | five regressors on the Cellpose-SAM encoder, central 512 x 512 crop | 0.25 |
| A | two whole-frame ViT-B/8 regressors (seeds 46, 47), four-flip test-time augmentation | 0.40 |
| D | one whole-frame DINOv3 ViT-B/16 regressor, epochs 16-20, four-flip test-time augmentation | 0.05 |
| V | two whole-frame DINOv3 ViT-L/16 regressors (seeds 46, 47), geometric mean, four-flip test-time augmentation | 0.30 |

After the blend:

1. Mask density. A zero-shot Cellpose-SAM segmentation gives a cell count per area. It enters with weight 0.1 when at least 50
   interior cells are found.
2. Stretch. A log-affine map with slope 1.075, fitted to minimise MAPE.

The V members are what p3b changes against tier p. They are trained on all labelled rows for 12 epochs from a self-supervised
initialisation, with an encoder learning rate of 2e-5 instead of 1e-5.

### CV

The base pipeline plus six log corrections: a DINOv3 ViT-B/16 term (0.125), three frozen token-statistic heads on the ViT-B/8 encoder
(spectrum 0.25, covariance square root 0.375, autocorrelation 0.25), and two quartets of graph networks over detected cell centres
(1.0 and 0.25).

### HEX

One nested ridge regression over 38 correction columns: cell detectors, a hexagonality corrector, graph networks, segmentation
features, and an attention-pooled DINOv3 ViT-B/16 column. A final step adjusts the value using cell-level features.

## Evaluation protocol

- Strict out-of-fold: every fitted quantity is refitted without the scored fold. The deployed parameters come from the same
  procedure run on all rows.
- Comparisons between stacks use the strict five-fold read and a repeated grouped 10-fold fit (20 repeats) with a donor-clustered
  bootstrap interval.
- Full-data members cannot be scored out of fold. Their fold-trained counterparts provide the out-of-fold predictions.

## Evidence for the ViT-L change

| Check | Result |
|---|---|
| Both seeds at 2e-5 against tier p, CD | -0.0258 strict (CI -0.040 to -0.012); -0.0306 on the 10-fold fit |
| One seed put back to 1e-5, CD | +0.0139 strict (CI +0.005 to +0.023); worse in every fold |
| Member alone, seed 47 at 2e-5 against 1e-5, CD | -0.023 (CI -0.075 to +0.028) |
| Full-data builds against their fold models, CD level | seed 46 -0.29 %, seed 47 +0.57 % |
| 46 labelled frames outside the whole-frame training set | no CD gap for the full-data whole-frame builds |
| Container output against the previous tier, 5 frames | CV and HEX equal within 1e-8 relative; CD differs only through V |

The gain in the stack is larger than the gain of the member alone.

## Limitations

- The run time on the challenge platform is unmeasured. The T4 estimate (258-296 s) is close to the 300 s limit.
- Submission 1 scored differently on the test set than out of fold: CD -0.062, CV -0.061, HEX +0.395. Whether this gap repeats for
  p3b is unknown.
- Eleven training rows have a CD error above 100 % and add 0.263 to the CD MAPE. They were left unchanged.
- Raising the encoder learning rate was not tested on the P1 slot. A ViT-L run at 3e-5 was stopped before it could be read.
