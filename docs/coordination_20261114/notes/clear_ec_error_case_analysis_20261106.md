# CLEAR-EC error-case analysis, 2026-11-06

Pod `clear-ec-research-4gpu-1104`, `experiments/error_case_analysis_20261106/`: `source/` (s00-s10, `common.py`), `logs/`, `tables/`, `sheets/`. CPU only, at most 4 workers. Reads `lane_gpu0_20261106/frame_stats.npz` and its `calib2` estimators read-only; nothing else touched. No commits.

Model: best OOF pipeline, 8954 rows, 6.264 / 10.151 / 9.803, mean 8.739. "Score" is the three-metric mean. All intervals are donor bootstrap unless stated. Source A rows count as singleton donors because `slide_id` equals the ID there, so A intervals are a little narrow.

## Conclusion

Do not build a set of image-noise augmentations. Error does not follow noise, sharpness, exposure, vignette or compression. Only CD follows one image trait, and that trait is how legible the cells are, which sensor-style noise does not explain. CV and HEX worst rows look like label scatter or region choice.

## 1. Share of error in the worst rows

Expected = same residual log SD, lognormal.

| Metric | Median APE | Worst 1 % holds (expected) | Worst 5 % | Worst 10 % |
|---|---:|---:|---:|---:|
| CD | 4.32 | 10.4 % (3.8) | 24.4 % (15.0) | 36.2 % (26.1) |
| CV | 8.09 | 6.5 % (4.0) | 18.0 % (15.4) | 29.1 % (26.5) |
| HEX | 8.11 | 5.2 % (4.0) | 17.2 % (15.3) | 28.3 % (26.5) |

Only CD has a real tail. Overlap of worst 5 % between metrics is 14-19 %. The worst 5 % sits in the outer label deciles: CD lowest decile 51 %; CV outer two 86 %; HEX lowest 72 %, over-predicted 86 %.

## 2. Traits against error

Bins: trait decile within source. Adjusted = minus the mean of source x label decile. Gain = change if the worst decile fell to the median bin, summed over the three metrics / 3. Shuffled = trait permuted within strata and the worst bin picked again, which is the floor of this statistic.

| Family | Representative trait | Gain on score | Shuffled median | Net |
|---|---|---:|---:|---:|
| Noise, calibrated | `sigma_floor` | -0.097 | -0.081 | -0.016 |
| Noise, other | `noise_imm`, `hf_floor` | -0.085, -0.099 | -0.08 | -0.008, -0.017 |
| Sharpness, calibrated | `focus2` | -0.123 | -0.080 | -0.043 |
| Sharpness, other | `lap_var`, `tenengrad` | -0.081, -0.090 | -0.08 | 0, -0.010 |
| Exposure | `mean`, `p50` | -0.102, -0.084 | -0.08 | -0.023, -0.005 |
| Contrast | `contrast` | -0.112 | -0.080 | -0.032 |
| Vignette | `center_corner`, worst bin is the flat frames | -0.147 | -0.078 | -0.068 |
| Compression, banding | `blk8`, `col_std`, `interlace` | -0.117, -0.108, -0.083 | -0.08 | -0.036, -0.029, -0.004 |
| Saturation, bubble glare | `frac255` | -0.077 | -0.051 | -0.026 |
| Detector count | `pk_n50` (CD -0.504, CI -0.706, -0.342) | -0.264 | -0.075 | -0.189 |
| Detector confidence | `pk_conf_med` | -0.218 | -0.075 | -0.142 |

- `blur2` was dropped as a sharpness read: synthetic blur sigma 1 moved it by -2.56, so it follows the noise floor. `focus2` and `sigma_floor` respond correctly (lane GPU 0 calibration). The blur2 rows in the logs should not be cited.
- Natural white-noise floor is narrow: A 2.43-4.29, B 2.00-2.92 (p1-p99). Focus sigma squared spans 3.7-7.7. Added sigma 6 gives a floor near 6.6, outside the data.
- Source A by `sigma_floor` band, adjusted APE: highest band (4.0+, n 253) CD +0.46 (-0.25, +1.25), CV +0.13 (-0.56, +0.91), HEX +0.40 (-0.34, +1.20).
- Bubble frames (> 0.2 % saturated, n 942): adjusted APE -0.12, +0.26, +0.18, all CI include 0.
- 2023-24 dark vignetted frames are not the worst: A 2023 8.85, A 2024 8.93, A 2022 9.91, A overall about 9.0.

Joint, out of fold: traits predicting adjusted APE.

| Set | R2 CD / CV / HEX | Gain metric CD / CV / HEX | Floor of statistic (random deciles) |
|---|---|---|---|
| All photometric | -0.002 / -0.014 / -0.008 | -0.294 / -0.357 / -0.083 | -0.138 / -0.213 / -0.081 |
| Noise only | -0.015 / -0.008 / -0.007 | -0.137 / -0.301 / -0.073 | same |
| Sharpness only | -0.011 / -0.019 / -0.004 | -0.134 / -0.201 / -0.048 | same |
| Detector | +0.028 / -0.020 / -0.011 | -0.794 / -0.085 / -0.146 | same |

Net of floor, all photometric is about -0.10 on the score if the association were fully causal and fully fixable. Noise and sharpness together are about 0 to -0.03.

Signed bias:

- Within label strata, detector count shows CV +7.7 % to -5.9 % and HEX -8.4 % to +5.3 % from its lowest to its highest decile. This is what a calibrated model does when the trait also predicts the label.
- Inside source x prediction decile it disappears: CV +0.1 (-1.5, +1.9), HEX 0.0 (-1.2, +1.2). All 11 photometric traits tested this way have a top-bottom bias CI that includes 0 (nine in `logs/s03.log`, `sigma_floor` and `focus2` in `logs/s09.log`). So there is nothing to correct.
- Nested correction from photometric traits: +0.075 (CI +0.057, +0.094), half strength +0.013. Worse.
- Nested multiplicative shift by legibility band: +0.0115 (CI +0.005, +0.018). Worse.

## 3. Natural experiment: fellow eyes, source B, 2184 pairs

Same donor, trait difference per within-source SD, APE adjusted for label stratum.

| Trait | dAPE CD | dAPE CV | dAPE HEX | d log pred, label-matched pairs, CD / CV / HEX |
|---|---|---|---|---|
| `sigma_floor` | 0.00 (-0.34, +0.34) | +0.52 (-0.06, +1.32) | +0.13 (-0.33, +0.62) | +0.2 / -0.4 / +0.7 % |
| `focus2` | +0.04 (-0.23, +0.34) | -0.59 (-2.02, +0.30) | -0.13 (-0.46, +0.22) | 0.0 / 0.0 / 0.0 % |
| `pk_conf_med` | -1.17 (-1.59, -0.74) | -0.45 (-0.98, +0.09) | +0.22 (-0.33, +0.73) | +0.6 / -3.3 / +4.5 % |

- CD, two traits together: confidence -1.18 (-1.59, -0.76) with `sigma_floor` +0.10 (-0.21, +0.42); confidence -1.19 with `focus2` +0.10 (-0.17, +0.39); confidence -1.12 with `center_corner` -0.18 (-0.48, +0.07).
- Lab response of a trained member (ledger S11): noise sigma 6 CD -6.8 %, blur sigma 1 CV +6.8 %. Natural differences do not move predictions anywhere near that. Those perturbations leave the range of the data.
- Still correlation: fellow eyes share a donor and probably a session, but the eye with lower confidence may also be the sicker eye.

## 4. Looking at the frames

Blind: 93 frames, 48 worst and 45 matched controls (same source and label decile, APE below median), shuffled, one rater (this session), scored before opening the key. L 3 = clear mosaic over most of the frame, 0 = none readable. Thumbnails at 0.37x, so pixel noise is not judged. `source/blind_scores.py`, `tables/blind_scored.json`, `sheets/blind_*.jpg`.

| Metric | Mean L worst / control | MWU p | L <= 1 worst / control |
|---|---:|---:|---:|
| CD | 0.94 / 2.00 | 0.006 | 69 % / 25 % |
| CV | 1.56 / 1.69 | 0.75 | 50 % / 46 % |
| HEX | 1.75 / 0.81 | 0.012 | 38 % / 81 % |

- L against `pk_conf_med` rho 0.81, `pk_n50` 0.82, `noise_imm` 0.11. So detector count and confidence do read legibility, and noise does not.
- What degrades these frames: haze, folds, bright flakes or granular damaged cells, dark patches, partial field, air bubbles. Thumbnails cannot show pixel noise, so 12 frames were also viewed at 1:1 (`sheets/noise_crops_1to1.jpg`): six of the highest `noise_imm` show readable mosaics or haze, with APE no worse than the six lowest. `noise_imm` follows `sigma_floor` only at Spearman 0.25-0.38, so it reads more than white noise; `sigma_floor` is the better noise read.
- HEX worst are clearer than their controls: low-HEX labels on clear regular mosaics, while low-HEX controls are frames the model reads as damaged. Legibility is a real HEX and CV cue (absolute Spearman with the label 0.33-0.47, within source), and the model uses it about as a calibrated model should (0.62-0.80 with the prediction, given r near 0.57 between prediction and label; figures are for detector confidence). An augmentation hiding it could cost CV and HEX.

CD top rows (`sheets/CD_worst_top12.jpg`):

- 2020-0644 and 2020-0651, both eyes: clear dense mosaic, labels 718-926, prediction and detector read about 2700-3000. Label is about 1/3.7 of what is visible. Nearby serials, so possibly a batch fault (unverified).
- 2020-0646 both eyes, 2021-0791: featureless fog, labels 372-896.

CV top rows: 2019-0064 both eyes, labels 0.03 and 0.11 with HEX 0; 0706-21 label 2.49 on a clear mosaic.

Frames with no usable window, 118 (1.3 %): APE 27.3 / 26.8 / 15.4, 5.7 / 3.5 / 2.1 % of total error, score -0.19 if they fell to the metric mean. Sheet: no readable cells in nearly all. Nothing to recover from these frames.

## 5. Model-fixable against label or region

Worst 5 % per metric, 448 rows. "Two reads" = best prediction and an OOF ridge on detector window statistics. Bound = change if those rows fell to the metric's median APE.

| Class | CD share, bound | CV share, bound | HEX share, bound |
|---|---:|---:|---:|
| Unreadable by detector | 26.6 %, -0.161 | 12.9 %, -0.118 | 11.4 %, -0.055 |
| Two reads agree, label away | 50.2 %, -0.197 | 83.3 %, -0.341 | 84.4 %, -0.357 |
| Detector near label, model away | 10.7 %, -0.033 | 0 %, 0 | 0.4 %, -0.001 |
| Neither near label, reads disagree | 12.5 %, -0.047 | 3.8 %, -0.016 | 3.8 %, -0.015 |

Caveat: the two reads share detector input and both shrink, so "agree" alone does not prove the label wrong. Independent evidence, source B:

| Metric | Label to fellow-eye label, worst / control | Prediction closer to fellow than own label, worst / control |
|---|---:|---:|
| CD | 0.119 / 0.063 | 41.5 % / 48.8 % |
| CV | 0.266 / 0.105 | 67.9 % / 58.7 % |
| HEX | 0.307 / 0.131 | 81.9 % / 69.6 % |

So CV and HEX worst rows are mostly one-eye label jumps the image does not show. Label inside the range of box-size windows on the same frame: CV 0.5 % (control 54 %), HEX 7 % (control 69 %), CD 15 % (control 23 %). So region choice, as far as detector windows can read it, does not reach these labels either. The windows overlap and carry detector error, so this is weak.

Bounds on labels that cannot be predicted:

- Rows with CD < 800, CV < 0.15 or CV > 0.9: 25 rows, score -0.18 if they fell to median.
- CD label / prediction below 0.5: 11 rows, 4.5 % of CD error, -0.091. Outside 0.8-1.25: 243 rows (2.7 %), 17.2 % of CD error, 74 % of them in the bottom quartile of `pk_n50`.
- One such row in the 1000 test images moves that submission's score by about 0.09, more than the 0.05 gap to first place. It hits every entry, so it matters for reading test against OOF.

Sampling floor from the brief, about 150 cells: SE 3 / 6.5 / 7.5 % against residual log SD 9.5 / 13.8 / 12.4 %.

Lead, not checked further, not noise: ID format `2019-NNNN-ODCN` (hyphen), 522 rows. CV trimmed MAPE 12.9 against 9.1 for space-format 2019, robust residual SD 0.159 against 0.116, best Spearman 0.35 against 0.48; worst-5 % rate on CV 14 % against 3.9 % for all space-format B. Median `noise_imm` and detector count are close (3.46 against 3.71, 754 against 693). Photometric traits identify the batch at AUC 0.84. Score bound -0.065 trimmed, -0.14 with its two extreme rows. Post hoc.

## 6. Augmentation families

Sizes are on the mean score, added to the pipeline. Except where a past run is cited they are judgement from sections 2-5, not measurements. Noise on the 1000-row test is about +-0.06.

| Family | Rating | Expected | Why |
|---|---|---:|---|
| Gaussian, shot, speckle noise | Skip | 0 to -0.01 | At the shuffled floor; natural range narrow; fellow eye 0 |
| Blur, defocus | Skip alone | 0 to -0.02 | Near the floor, fellow eye 0; CD low-focus decile +1.1 only |
| Gamma, brightness, contrast | Skip alone | 0 to -0.02 | Small and between donors only, fellow eye 0 |
| Vignette, illumination | Skip | 0 to -0.02 | Worst bin is the flat frames; gone once legibility is held fixed |
| JPEG blocks, banding, interlace | Skip | 0 | Near the floor |
| Bubble, fold, dark patch | Skip | 0 | Bubble excess CI includes 0; tags not enriched in worst rows |
| A to B style transfer | Skip | 0 or harmful | Source carries label (mean CV 0.418 vs 0.347) |
| Zoom with label rescale | Skip | 0 | Run: member +0.088 (CI 0.002, 0.169), added 0.000 |
| Measured bundle (focus, exposure, noise from a same-source frame): `measured_v1`, already queued on lane GPU 0 | Try one, let it finish | Member about -0.04, added 0 to -0.02 | Run of the older bundle: member -0.042 (CI -0.104, +0.019), added -0.005; fold 0 only, any gain would be regularisation and not the noise mechanism |
| Haze or border fade, CD head only | Try one, only if a GPU has nothing better | CD -0.05 to -0.15, score -0.02 to -0.05, likely 0 | Only trait with within-donor support. Cap -0.17 on the score, about half of it frames with no usable window, where there are no cells to read. Keep it out of CV and HEX |

What this analysis cannot show: a regulariser could lower error everywhere with no link to any trait. Only a training run shows that, which is why the bundle deserves its run. No test images were seen, so a noisier test set is not excluded; CD and CV test scores near OOF argue against it.
