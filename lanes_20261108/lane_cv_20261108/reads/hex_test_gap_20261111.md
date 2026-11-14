# Submission 1 test gap: can our OOF explain HEX +0.395 with CD and CV slightly better?

CV lane, 2026-11-11 (PI request via the coordinator). CPU only; no test images or labels used; no adjustment applied.
Script: `source/hex_test_gap_20261111.py` (`step1 | step2 | step2x | step3 | step4`), results in `reads/hex_test_gap_20261111.json`, logs `logs/hex_gap_step*.log`.

Observed (test minus OOF, submission 1 = hybrid B): CD −0.062, CV −0.061, HEX +0.395 (test 6.250 / 10.143 / 10.510; OOF 6.312 / 10.204 / 10.115).
Per-row OOF: `recipeA_analysis_20261028/hybrid_oof.npz`, key `B_hybrid_w0p5`; it reproduces the three OOF numbers exactly. 8954 frames, 6768 donors (`eval_oof_20261031.donors`, slide id), median 1 frame per donor.

## Answer first

1. **Yes: noise explains the pattern, and no shift is needed.**
   - As a three-metric vector, the test-minus-OOF gap sits at the 60th percentile of 1000-image sampling noise: Mahalanobis tail 0.41 for row-random draws and 0.40 for donor-clustered draws.
   - HEX alone is a 1-in-16 upper tail (z = +1.5). CD and CV are ordinary (z = −0.2 and −0.1).
   - Donor clustering changes nothing: 6768 donors, median 1 frame each.
2. **If it is not noise, two explanations fit. The test score cannot tell them apart.**
   - **(a) The test set's HEX-label mix has more frames at the extremes.** That is where our HEX predictions shrink toward the mean: HEX APE is 16–23 % in the end deciles against 6–8 % in the middle.
     - It is the only covariate that reproduces all three gaps with a small shift: KL 0.006, largest change ×1.27 (frames with HEX ≥ 0.66 go from 10.3 % to 13.0 %).
     - For it to be the whole explanation, the label mix must move 2.6–3.3 SD along that direction, measured against 1000-image sampling. That would be a real population difference, not sampling luck.
   - **(b) A HEX-only multiplicative label offset of about −3.4 % or +3.6 %.**
     - MAPE is almost symmetric in b, so the score cannot give the sign.
     - An offset shared by all three metrics does not fit. It would raise CD by about 0.76 and CV by about 0.40, which would put the observed CD gap at z ≈ −2.6.
   - **Nothing we can measure on the images explains it.** We checked image, segmentation, detector, source/ID-format and prediction covariates. Each one either cannot reproduce the three gaps at all, or needs whole deciles to nearly vanish (KL 0.63–0.88, 60–90 times the 99th percentile of ordinary 1000-image sampling).
     - Source mix in particular cannot move HEX: even an all-P2 test set gives only HEX +0.04.
   - **A deploy-side HEX offset is ruled out.** On the 60 smoke-test rows, deployed vs OOF hybrid HEX has a mean log ratio of −0.15 % (SD 1.0 %), not 3 %.
3. **No HEX adjustment is safe.**
   - **A scale fix (×0.9675 or ×1.0375) helps only if (b) is true and the sign is right.** Then it recovers 0.35–0.43.
     - With the wrong sign it costs +1.05 to +1.27. If the gap is noise it costs +0.35 to +0.44.
     - Even granting (b) with an unknown sign, the expected change is +0.35 to +0.42 (worse).
     - Among random draws that turn out this bad on HEX, a scale fix never helped (0 of 1801).
   - **The fix aimed at (a), a 1.16 stretch of HEX predictions, is worth at most −0.08.** That figure is in-sample.
   - **Doing nothing is the minimax choice.**

## 1. Noise scale (1000-image draws, 20 000 draws each)

| | CD | CV | HEX |
|---|---|---|---|
| SD of the gap, row-random | 0.318 | 0.518 | 0.257 |
| SD of the gap, donor-clustered | 0.388 | 0.589 | 0.258 |
| Observed gap, z (row / donor) | −0.20 / −0.17 | −0.13 / −0.11 | +1.54 / +1.53 |
| One-sided tail (row / donor) | 46 % / 51 % | 57 % / 61 % | 6.4 % / 6.5 % |

- Correlation between the metrics' gaps is only about 0.10 to 0.12, so a high HEX draw says almost nothing about CD and CV.
- Joint, as a vector: Mahalanobis d² = 2.57 (row-random) and 2.50 (donor-clustered) on 3 dimensions. The tail is 41 % and 40 %, so the gap vector sits inside the noise ellipse.
- Joint, as the literal pattern: "HEX ≥ +0.395 and CD ≤ −0.062 and CV ≤ −0.061" happens in 1.2 % (row-random) and 1.4 % (donor-clustered) of draws. That small number is mostly the HEX tail (6.4 %) times two roughly 50 % sign conditions. Nothing about CD or CV is unusual.
- Donor clustering changes nothing material. Most donors give one frame, so clustered and row-random draws are nearly the same draw. The donor-clustered draws hold a median of 1000 (5th–95th percentile 1000–1001) rows.

## 2. Composition search (minimum-KL exponential tilting)

Method:
- Every covariate is cut into 10 quantile bins, or its natural categories, plus a "missing" bin where needed.
- The tilt reweights the bins as q_b = p_b·exp(λ·A_b)/Z, where A_b is the bin's mean OOF APE per metric. λ is solved (dual, BFGS) so that the reweighted means hit all three test values at once.
- A covariate is called feasible if every metric is within 0.005 of its target.
- Reaching the target says nothing about plausibility by itself. So every required shift is compared with the KL that ordinary 1000-image sampling produces: for a random 1000-row histogram on 10 bins, the KL against the pool has a median of 0.0037–0.0043 and a 99th percentile of 0.0095–0.0106 across the covariates (4000 draws; the χ² approximation 9/2000 = 0.0045 agrees). The tilt KL is also a z-score along the tilt's own direction: the projection of the random histogram deviation onto log(q/p).

### Single covariates

Six covariates hit all three targets. Only one of them does it with a plausible shift:

| Covariate | KL needed | Largest changes | Plausible? |
|---|---|---|---|
| **label_HEX** | **0.0062** | HEX ≥ 0.66 ×1.27 (10.3 % → 13.0 %); HEX 0.54–0.57 ×0.91 (14.2 % → 12.9 %) | The size is ordinary (between the sampling median 0.004 and p99 0.0099). Along its own direction it is z = 3.3 (tail 0.03 %). |
| legible_fraction | 0.633 | three deciles ×0.02–0.04 (9.9 % → 0.2–0.3 %) | No: whole deciles have to vanish (60× the sampling p99) |
| pred_CD | 0.653 | CD-prediction deciles ≥ 3048 and 2651–2754 ×0.03–0.06 | No |
| seg_h6 | 0.662 | h6 deciles 0.546–0.598 and < 0.374 ×0.07–0.12 | No |
| det_f_active10 | 0.804 | three deciles ×0.01–0.10 | No |
| label_CD | 0.881 | CD-label deciles 2247–2481 and 2924–3135 ×0.03–0.11 | No |

The other covariates cannot reach all three targets even with extreme reweighting. The column below is the closest fit; the target is −0.062 / −0.061 / +0.395:

| Covariate | Best reachable CD / CV / HEX gap | HEX-only tilt (KL; what it implies for CD / CV) |
|---|---|---|
| pred_HEX | −0.165 / −0.129 / +0.255 | KL 0.29; lowest HEX-prediction decile ×3.6; CD +2.10, CV +1.08 |
| pred_CV | −0.282 / −0.344 / +0.160 | KL 0.53; CD +2.57, CV +1.23 |
| det_n_det | −0.075 / −0.316 / +0.153 | KL 0.22; fewest-detections decile ×3.2; CD +1.58, CV +1.12 |
| img_mean | −0.290 / −0.363 / +0.388 | KL 1.20 |
| img_contrast | −0.523 / −0.158 / +0.056 | KL 0.51; lowest-contrast decile ×4.7 (10 % → 47 %); CD +0.38, CV +0.96 |
| img_focus | −0.813 / −0.274 / +0.375 | KL 2.30 (degenerate) |
| img_black_frac | −0.751 / +0.031 / +0.004 | cannot move HEX |
| source_P1_P2 | −0.815 / +0.045 / +0.040 (100 % P2) | cannot move HEX: the most it gets is +0.04 with an all-P2 test set |
| id_format | −0.815 / −0.789 / −0.081 | KL 0.62; P2-hyphen IDs ×7.7 (5.8 % → 45 %); CV +2.61 |
| seg_n_int, seg_area_cv, seg_cd_mm2, seg_coverage | HEX reaches only −0.33 to +0.08, with CD / CV far off | KL 0.13 each, all through the 6 % of frames with no segmentation features (×3.3–3.4); CD +1.1 to +1.3, CV +0.76 to +0.84 |
| label_CV | +5.3 / +7.9 / +2.8 (diverges) | KL 0.047; CV labels ≥ 0.47 ×1.8; CD +0.55, CV +1.34 |

Reading: no measurable image, segmentation, detector or source property of the frames explains HEX up while CD and CV go down. Every HEX-only tilt on a non-label covariate pushes CV up by +0.41 to +2.61, and CD up by +0.38 to +2.57 (the one exception is id_format, CD −0.31 with CV +2.61). That contradicts the observed CD and CV. The one route that works runs through the HEX labels themselves, and those are not observable on the test set.

### Pairs of the best three (5 × 5 bins)

The best three (lowest KL among the feasible) are label_HEX, legible_fraction and pred_CD.

| Pair | KL | Reading |
|---|---|---|
| label_HEX × legible_fraction | 0.0044 | Feasible. Essentially the label_HEX shift; legible fraction adds almost nothing (largest cell ×1.28). |
| label_HEX × pred_CD | 0.0064 | Feasible. Again the label_HEX shift (HEX ≥ 0.62 with mid CD predictions ×1.20). |
| legible_fraction × pred_CD | 0.141 | Reaches the targets only by inflating 0.1 % cells ×26–39. At 25 cells the sampling median is about 0.012 (24/2000), so this is not plausible. |

Pairs add nothing beyond label_HEX alone.

## 3. Two hypotheses

### (a) HEX label shift (the test set's HEX-label mix differs)

Tilt of the HEX-label decile mix that produces exactly +0.395 on HEX (HEX-only target):

| | Pool (OOF) | HEX-only tilt | All-three tilt (also matches CD, CV) |
|---|---|---|---|
| KL | – | 0.0029 | 0.0062 |
| share HEX < 0.45 | 14.2 % | 16.2 % (×1.14) | 14.7 % |
| share HEX ≥ 0.66 | 10.3 % | 11.2 % (×1.09) | 13.0 % (×1.27) |
| mean HEX label | 0.546 | 0.544 | 0.551 |
| SD of HEX label | 0.085 | 0.089 | 0.090 |
| implied CD / CV gap | – | +0.088 / +0.110 | −0.062 / −0.061 (by construction) |
| z along its direction vs 1000-image sampling | – | 2.6 (tail 0.4 %) | 3.3 (tail 0.03 %) |

- The shift widens the HEX-label distribution: more frames at both ends, mean unchanged. Its cost comes from regression to the mean.
  - OOF HEX APE by HEX-label decile is 22.8 % for HEX < 0.44, 16.3 % for HEX ≥ 0.66, and 5.7–7.6 % in between.
- Implications for CD and CV: a pure HEX label shift predicts CD and CV slightly worse (+0.09, +0.11), while they came out slightly better. The difference (−0.15, −0.17) is 0.5 SD and 0.3 SD of the noise, so (a) is consistent with the observed CD and CV.
- As the only cause, (a) needs a label-mix move of z 2.6 along its direction. That is more than the z 1.5 that noise alone needs, because label composition carries only part of the HEX APE variance.
  - (a) is therefore a real difference in the test population if true.
  - A mix of a smaller label shift and ordinary noise is the more natural version of it. It was not fitted separately.

### (b) Multiplicative offset b (test labels = y·(1 + b))

ΔMAPE(b) relative to b = 0, on OOF:

| b | −10 % | −7.5 % | −5 % | −3.25 % | −2 % | −1 % | 0 | +1 % | +2 % | +3.75 % | +5 % | +7.5 % | +10 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CD | 5.93 | 3.51 | 1.62 | 0.69 | 0.26 | 0.06 | 0 | 0.06 | 0.25 | 0.81 | 1.38 | 2.78 | 4.37 |
| CV | 3.61 | 1.99 | 0.87 | 0.36 | 0.14 | 0.04 | 0 | 0.03 | 0.12 | 0.42 | 0.74 | 1.60 | 2.69 |
| HEX | 3.67 | 2.02 | 0.87 | 0.353 | 0.12 | 0.03 | 0 | 0.03 | 0.13 | 0.431 | 0.75 | 1.57 | 2.60 |

- The HEX curve crosses +0.395 at b ≈ **−3.4 %** and **+3.6 %**, found by interpolating between grid points. The grid points used below, −3.25 % and +3.75 %, give +0.353 and +0.431.
- The curve is almost symmetric, so the score cannot say whether the test HEX labels are lower or higher than ours.
- CD and CV are at their minimum at b = 0. Our predictions are MAPE-calibrated, so any offset only hurts.
- An offset shared by all three metrics does not fit the observed pattern. At ±3.5 % it predicts CD +0.75 to +0.77 and CV +0.39 to +0.41, which puts the observed CD gap (−0.062) at z ≈ −2.6 for row-random draws (−2.1 donor-clustered) and CV at z ≈ −0.9.
- A HEX-only offset fits all three by construction: CD and CV are untouched, and their gaps are ordinary noise.
- Deploy-side check: on the 60 fold-4 smoke-test rows (`submission1_test_20261030/n60/result.json`), deployed hybrid HEX vs OOF hybrid HEX has a mean log ratio of −0.15 % (median −0.11 %, SD 1.0 %, SE ≈ 0.13 %). CD is −0.5 % and CV +0.08 %.
  - The deployed pipeline therefore does not carry a 3 % HEX offset relative to the OOF.
  - Caveat: the recipe A half of the deployed hybrid is full-data and in-sample for these rows.
  - An offset, if real, would have to come from the test labels (a convention or device difference), not from our pipeline.

## 4. Decision table (computed on OOF; nothing applied)

Each cell is the HEX MAPE change versus no adjustment, if the column's hypothesis is true. CD and CV are untouched in every row.

| HEX adjustment | H0 noise (base 10.115) | H0 given this gap (random draws with HEX gap 0.395 ± 0.05; n = 1801) | Offset b = −3.25 % (base 10.468) | Offset b = +3.75 % (base 10.546) | (a) label mix (base 10.510) |
|---|---|---|---|---|---|
| ×0.9675 (fixes b < 0) | +0.347 | +0.302 (helps in 0 %) | **−0.353** | +1.045 | +0.431 |
| ×1.0375 (fixes b > 0) | +0.442 | +0.454 (helps in 0 %) | +1.266 | **−0.431** | +0.297 |
| stretch g = 1.16, scale 1.005 (fit to (a) on tilted OOF; in-sample) | +0.013 | +0.011 (helps in 41 %) | +0.144 | −0.107 | **−0.083** |

- **Offset fix:** it only pays if the sign is known.
  - With the sign unknown and the offset taken as certain, the expected change is +0.35 (×0.9675) and +0.42 (×1.0375).
  - Under noise, a random 1000-image draw that came out this bad on HEX is not bad in one direction. A global scale fix never helped in 1801 such draws, nor in the 500 that also had CD and CV ≤ 0.
- **Label-mix fix (wider HEX predictions):** small stakes either way. It gains at most 0.08, which is an in-sample upper bound, and costs 0.01 under noise. Its effect under an offset depends on the sign.
- **No adjustment:** the minimax choice. It is never more than 0.43 worse than the best adjustment under any hypothesis, and that 0.43 is only reachable if one knows which offset sign holds.

As requested, nothing above was applied to any submission or deploy file.

## Caveats

- The tilts reweight our OOF rows. They cannot create frames unlike any in the training pool, and a test shift along an unmeasured covariate would show up here as noise.
- The offset curves treat an offset as exact and uniform. A partial or frame-dependent offset sits between the offset and noise columns.
- Whether the final evaluation reuses these same 1000 test images decides whether "noise" reverts. If it reuses them, the bad HEX draw is a fixed property of the test set, and the "H0 given the gap" rows are the relevant ones.
