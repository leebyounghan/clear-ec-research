# Lane GPU 3: other model families at the winner's budget (idea G4)

Reads follow `protocol.json`. Reference: recipe A (CPDINO ViT-B, kernel 16, stride 8), same folds, seed 46. Mean of epochs 16-20, paired on the same frames, CI from donor bootstrap. Lower is better.

## Reads

| Arm | Folds, seed | Arm | Reference | Delta (CI) | Added, w 0.125 (CI) | Added, w 0.25 (CI) | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| DINOv3 ViT-B/16, stride 16 | 0, 4; 46 | 9.320 | 9.046 | +0.274 (+0.185, +0.361) | -0.0165 (-0.0284, -0.0043) | -0.0123 (-0.0363, +0.0118) | Fails both gates (member <= -0.05; added <= -0.02, CI below 0) |
| ConvNeXt base, stride 32 | 0, 4; 46 | 9.632 | 9.046 | +0.586 (+0.478, +0.694) | -0.0016 (-0.0167, +0.0132) | +0.0292 (-0.0004, +0.0584) | Fails both gates |
| DINOv2 ViT-B/14 | 0, 4; 46 | 9.500 | 9.046 | +0.454 (+0.369, +0.541) | -0.0053 (-0.0174, +0.0066) | +0.0127 (-0.0112, +0.0364) | Fails both gates |
| The three above, equal-weight log mean | 0, 4; 46 | 9.145 | 9.020 | +0.125 (+0.046, +0.202) | -0.0136 (-0.0243, -0.0031) | -0.0134 (-0.0345, +0.0075) | Fails the added gate (<= -0.02); not better than DINOv3 alone |

The combined row compares five-epoch log means on both sides, so its figures are not on the same footing as the single-arm rows. Details in `reads/<arm>_s46.json`. Added reads are on the rows of the folds read; the best pipeline scores 8.7578 on these rows.

## DINOv3 ViT-B/16, stride 16

| Fold | Arm | Reference | Delta | CD | CV | HEX | Residual r with reference, CD / CV / HEX | Train probe, epoch 20, arm / reference |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 9.375 | 9.081 | +0.294 | +0.341 | +0.254 | +0.288 | 0.91 / 0.94 / 0.95 | 7.51 / 8.30 |
| 4 | 9.265 | 9.012 | +0.253 | +0.188 | +0.308 | +0.264 | 0.90 / 0.94 / 0.95 | 7.66 / 8.62 |

- Added at w 0.125, per metric: CD -0.015, CV -0.027, HEX -0.008. With each of epochs 16-20 alone: -0.016 (CI -0.028, -0.004).
- Epochs 11-15 are no better for the arm relative to the reference (+0.285, +0.207), so the fixed last-five read is not what hurts it.
- The winner's budget did not close the gap: the older run with no SSL and mean pooling was +0.237 on fold 3 (9.539 against 9.302), and the older added reads were -0.005 to -0.020 at 20 %. Different folds, one seed each.
- The arm fits training frames better than the reference and validates worse. This is one family at half the token density; `dinov3_b16_s8` separates density from pretraining.
- Not run on seed 47: the protocol confirms arms that pass.
- Cost: 175 s per epoch, 1.56 h per fold including SSL, on one H200 without keepalive.

## ConvNeXt base, stride 32

| Fold | Arm | Reference | Delta | CD | CV | HEX | Residual r with reference, CD / CV / HEX | Train probe, epoch 20, arm / reference |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 9.668 | 9.081 | +0.588 | +0.337 | +0.804 | +0.622 | 0.88 / 0.92 / 0.93 | 6.75 / 8.30 |
| 4 | 9.596 | 9.012 | +0.584 | +0.443 | +0.715 | +0.595 | 0.87 / 0.91 / 0.93 | 6.51 / 8.62 |

- Pooled per metric: CD +0.390, CV +0.759, HEX +0.609.
- Added at w 0.125, per metric: CD -0.021, CV 0.000, HEX +0.016. With each of epochs 16-20 alone: -0.0008 (CI -0.016, +0.014). At w 0.25 the added member hurts: +0.029, CD -0.009, CV +0.039, HEX +0.058.
- The fixed last-five read costs this arm, and it does not change the verdict. Validation falls to 9.284 at epoch 5 on fold 0 and 9.331 at epoch 8 on fold 4, then rises to 9.733 and 9.658 as the train probe reaches 6.75 and 6.51. Those best single epochs, chosen on validation, are still +0.20 and +0.32 above the reference.
- The grid is 31x41 tokens against 124x164 for the reference. Its residuals are the least correlated with the reference of the arms read so far, and that still gives no gain.
- Not run on seed 47: the protocol confirms arms that pass.
- Cost: 194 s per epoch, about 1.8 h per fold including SSL, 51 GiB at micro batch 16.

## DINOv2 ViT-B/14

Fold 0 ran on GPU 1 and fold 4 on GPU 3, both with this lane's runner and roots.

| Fold | Arm | Reference | Delta | CD | CV | HEX | Residual r with reference, CD / CV / HEX | Train probe, epoch 20, arm / reference |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 9.573 | 9.081 | +0.492 | +0.660 | +0.509 | +0.308 | 0.88 / 0.95 / 0.96 | 7.80 / 8.30 |
| 4 | 9.426 | 9.012 | +0.415 | +0.545 | +0.536 | +0.164 | 0.87 / 0.95 / 0.96 | 8.25 / 8.62 |

- Pooled per metric: CD +0.602, CV +0.523, HEX +0.236.
- Added at w 0.125, per metric: CD 0.000, CV -0.001, HEX -0.015. With each of epochs 16-20 alone: -0.0035 (CI -0.016, +0.008).
- The best single epochs, chosen on validation, are 9.415 at epoch 12 on fold 0 and 9.288 at epoch 10 on fold 4, still +0.33 and +0.28 above the reference.
- Token density alone does not order the three families: DINOv2's 70x93 grid is denser than DINOv3's 62x82 at stride 16 and it scores worse. One seed each. `dinov3_b16_s8` tests density directly.
- Not run on seed 47: the protocol confirms arms that pass.
- Cost: about 200 s per epoch, about 1.8 h per fold including SSL.

## The three families as one added member

Gate written in `protocol.json` before DINOv2's two-fold read existed. No selection among the arms. Read: `reads/combined_dinov3_convnext_dinov2_s46.json`, tool `source/read_combined_20261107.py`.

- Added at w 0.125: -0.0136 (CI -0.0243, -0.0031), CD -0.020, CV -0.014, HEX -0.007. With epochs 16-20 one at a time: -0.0130 (CI -0.0237, -0.0025). The CI excludes 0 and the effect is below the -0.02 gate.
- Added at w 0.25: -0.0134 (CI -0.0345, +0.0075). At w 0.5 it hurts: +0.040 (CI -0.001, +0.081).
- DINOv3 alone at w 0.125 is -0.0165 (CI -0.0284, -0.0043), so the other two families do not add to it.
- The three-family member alone is 9.145 against 9.020 for recipe A's five-epoch log mean. Residual r with the reference: CD 0.92, CV 0.96, HEX 0.97.
- Seed 46 only, on the rows of folds 0 and 4. No seed 47 run follows, since the gate was not met.

## Lane GPU 1's end-to-end job, run on GPU 3

SSL arm `cropopt_both` (crop 544, broad optics, mean and std), recipe A, seed 46. Fold 0 ran on GPU 3 and finished 2026-11-07 07:04 EST; fold 4 ran on GPU 1. Both `exit.json` completed. Numbers from `reads/e2e_cropopt_both_s46.json`, by this lane's read tool. Lane GPU 1 owns the verdict under its protocol, which has the same gates.

| Fold | Arm | Reference | Delta | CD | CV | HEX | Residual r with reference, CD / CV / HEX |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 9.114 | 9.081 | +0.034 | +0.069 | +0.001 | +0.030 | 0.98 / 0.99 / 0.99 |
| 4 | 9.015 | 9.012 | +0.004 | +0.022 | +0.016 | -0.028 | 0.98 / 0.99 / 0.99 |

- Two folds, pooled: +0.0185 (CI -0.020, +0.058), CD +0.046, CV +0.009, HEX +0.001. Gate <= -0.05: not met.
- Added to the best pipeline (8.7578 on these rows): -0.0038 at w 0.125 (CI -0.0097, +0.0017), -0.0030 at w 0.25 (CI -0.0151, +0.0081). Gate <= -0.02, CI below 0: not met.
- One seed.

## Queue

`dinov3_b16_s8` fold 0 stopped by verified pid on 2026-11-06 23:12:57 EST, after one regression epoch; its SSL is kept and the partial regression is in `runs/dinov3_b16_s8_f0_s46/stopped_regression_20261107`. Lane GPU 2 runs `dinov3_b16_s8` folds 0 and 4 on GPU 2 with this lane's runner and roots, and `dinov3_b16` seed 47 in its own folder. Lane GPU 1 runs `cpsam_c512` fold 4 on GPU 1 with this lane's root, since 2026-11-07 10:57 EST. This lane's queue: `cpsam_c512` fold 0, running since 2026-11-07 12:29 EST. `dinov3_l16` waits on the PI's approval of the weights download. CPSAM on the whole frame is dropped on measured cost, see `protocol.json`.
