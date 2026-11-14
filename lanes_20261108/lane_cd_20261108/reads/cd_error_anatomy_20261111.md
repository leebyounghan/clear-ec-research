# CD error anatomy of the frozen candidate (D-CDANAT-001, 2026-11-11 22:30 EST)

**Answer: no.** No untried cheap lever is worth more than 0.03 on CD. The four levers asked about (a global or per-level multiplier, clipping, a disagreement shift, mask density at full weight on frames with many cells) all make the strict value worse or leave it unchanged. The one large block of error that is not spread over every row is about 6 frames whose labels contradict the image (0.19 MAPE points). Those frames cannot be predicted.

Base: D-CD-001 strict five-fold OOF, **6.1743** (`reads/strict_nesting_20261111.npz` `v3_strict`). Nothing in the freeze was changed. Code: `source/cd_error_anatomy_20261111.py` (parts 1, 2, 4) and `source/cd_levers_20261111.py` (part 3). Numbers: `reads/cd_error_anatomy_20261111.json` and `reads/cd_levers_20261111.json`. MAPE in %.

## 1. Where the error sits

| Slice | Rows | CD MAPE | Share of total error |
|---|---:|---:|---:|
| P1 (old-format ids) | 4,514 | **7.04** | 57.5 % |
| P2 (YYYY-NNNN ids) | 4,440 | **5.29** | 42.5 % |
| Label decile 1 (CD < 2,250) | 889 | 13.14 | 21.1 % |
| Label deciles 2-9 | 7,167 | 4.25-6.46 | 65 % |
| Label decile 10 (CD >= 3,310) | 898 | 8.45 | 13.7 % |
| Interior cells (seg_loose n_int) 1-49 | 715 | 8.96 | 11.6 % |
| n_int 50-149 | 952 | 7.4-7.8 | 13.0 % |
| n_int 150-249 | 776 | 6.56 | 9.2 % |
| n_int >= 250 | 5,976 | 4.95 | 53.5 % |
| Mask coverage, lowest fifth | 1,684 | 7.66 | 23.3 % |
| Mask coverage, highest fifth | 1,684 | 4.18 | 12.7 % |
| Noise floor (sigma_floor), two noisiest fifths | 3,582 | 6.75-7.22 | 45 % |
| Noise floor, three cleanest fifths | 5,372 | 5.57-5.72 | 55 % |
| Focus (focus2), softest fifth | 1,791 | 7.07 | 22.9 % |
| Focus, sharpest fifth | 1,791 | 5.45 | 17.6 % |

- **The error follows legibility and the low end of the label range.** Fewer interior cells, lower mask coverage, softer focus and labels below 2,250 all raise MAPE. Detector confidence and detector peak counts are flat (MAPE 5.8-6.5 in every fifth).
- **Bias by label level is what MAPE-optimal shrinkage produces:** low labels are over-predicted (median log ratio +0.035 in decile 1) and high labels under-predicted (-0.038 in decile 9, -0.071 in decile 10). Both ends carry the error: decile 1 at 13.1 and decile 10 at 8.5, against 4.3-6.5 in between. Part 3 shows that correcting it per predicted level does not help.
- **Tail:**

| Worst rows | Rows | Share of total APE | CD if they were at the median APE | P1 share | Labels off the calibrated mask CD by > 2x | KG impossible serials (2020-0644..0651) | HEX-zero rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1 % | 9 | 4.0 % | 5.93 (-0.245) | 11 % | 5 | 6 | 0 |
| 1 % | 90 | 10.4 % | 5.58 (-0.597) | 62 % | 5 | 6 | 1 |
| 5 % | 448 | 24.5 % | 4.87 (-1.30) | 74 % | 5 | 6 | 1 |

- The worst 0.1 % is mostly the KG's impossible-label frames (`find-impossible-labels`): labels 372-926 where the image reads about 2,700. They are all over-predicted, and those 6 rows alone are worth **0.187** MAPE points.
- Only 6 rows in all have a label more than 2x away from the calibrated mask CD (rows with >= 50 interior cells), worth 0.13. Beyond those rows the tail is spread over ordinary frames: in the worst 5 %, the median label is 2,415, the median prediction 2,601, and 208 of 448 are under-predicted. Removing a tail would require knowing which rows to remove; no input flags them.

## 2. Why P1 is harder

| | P1 | P2 |
|---|---:|---:|
| Label quantiles 5 / 25 / 50 / 75 / 95 % | 2,113 / 2,732 / 2,954 / 3,175 / 3,571 | 1,953 / 2,433 / 2,695 / 2,941 / 3,300 |
| CD MAPE | 7.04 | 5.29 |
| Same source, reweighted to the other source's label mix | P1 at P2's mix: **8.51** | P2 at P1's mix: **5.10** |
| Median interior cells / share with >= 50 | 347 / 80 % | 671 / 92 % |
| MAPE on frames with >= 150 interior cells | 5.80 | 4.60 |
| Noise floor median (sigma_floor) / focus2 median | **3.58** / 5.45 | **2.53** / 5.43 |
| Within-donor groups (rows) | 1,414 donors, median 3 frames (4,073 rows) | 2,186 donors, mostly the two fellow eyes (4,373 rows) |
| Within-donor SD of log label | **0.137** | **0.058** |
| Within-donor SD of log prediction | 0.117 | 0.055 |
| Within-donor SD of the log residual | **0.083** | **0.043** |
| Correlation of within-donor label and prediction moves | 0.80 | 0.71 |

- **Not the label range.** P1 is harder at every label level: at P2's label mix it gets worse (8.51), and P2 at P1's mix gets slightly better (5.10). P1's bottom quintile is the worst cell in the table (12.0 vs 8.8 for P2's bottom quintile).
- **Partly image quality.** P1 frames are noisier (noise floor 3.58 vs 2.53, the same focus) and less legible (half as many interior cells), but P1 is still worse among frames with >= 150 interior cells (5.80 vs 4.60).
- **Mostly label scatter within a donor.** P1 labels move 2.4 times as much between frames of the same donor as P2 labels do between the two eyes. The predictions follow those moves (r 0.80), but the within-donor residual is still twice P2's (0.083 vs 0.043 in log). If the frames of a P1 donor are the same eye, much of that residual is label noise. The ids do not say which eye, so this is a proxy, not a measurement.

## 3. Easy levers (nested five-fold against 6.1743; one form each, fixed before the read; the correction is fit on the other four folds' strict OOF and applied to the fifth)

| Lever | Status | CD | Change | Per fold |
|---|---|---:|---:|---|
| (a1) global MAPE-optimal multiplier | new | 6.1789 | **+0.0046** | +0.003 / +0.002 / -0.000 / +0.006 / +0.013; fitted multipliers 0.9983-1.0008 |
| (a2) per-predicted-level multiplier (5 equal-count bins) | new | 6.1793 | **+0.0050** | -0.000 / +0.003 / -0.011 / +0.012 / +0.021; multipliers 0.994-1.005 |
| (b) clip to the training folds' label 0.5 / 99.5 % | new | 6.2127 | **+0.0384** | worse in every fold (it moves only the 2-17 rows per fold outside the range, and those labels really are extreme) |
| (c) lower prediction on frames with high member disagreement | **closed** (KG `exp-cd-uncertainty-shift`, `idea-mape-shift-predicted-noise`) | 6.215-6.228 on the 6.204 base | **+0.012 to +0.025** | k chosen near 0; third version after perturbation sensitivity and legibility bands. Not rerun. |
| (d) calibrated mask CD at full weight on frames with >= 150 interior cells | new | 7.5238 | **+1.35** | worse in every fold. On those 6,752 frames the stack's MAPE is 5.13; the mask alone is far worse. Weight 0.1 (in the stack) is right. |
| Mask-density variants (median, trimmed, band, densest window, >= 30 / >= 100 cell rules, rotated segmentation) | **closed** (KG `exp-cd-mask-density-variants`) | 6.186-6.195 on the 6.204 base | -0.009 to -0.018 = the plain mean, already in the stack | not rerun |
| Per-source stretch | **closed** (KG `exp-cd-per-source-stretch`; per-source choices also break the charter) | 6.255 (best pipeline base) | -0.009 on source B | not rerun |

The stretch inside the stack already takes the global and level-wise MAPE optimum: the second-level multipliers fit to 1.000 ± 0.005 and only add fold noise.

## 4. Bounds

- **If P1 had P2's error:** CD 5.29 (P2's own MAPE), -0.88. Part 2 shows the P1 excess comes from noisier, less legible frames and label scatter, not from a fixable offset; a per-source correction is closed above.
- **150-cell sampling floor:** sqrt(2/pi) x CV / sqrt(150) per frame, with the frame's CV label (median 0.37): mean **2.49 %** (P1 2.72, P2 2.25). A label from 150 cells carries this error even with perfect cell reading. Device behaviour and other label noise come on top (root-cause analysis 11-09: the pipeline already beats expert centres on CD).
- **Test noise:** the SD of a 1,000-row CD mean is 0.318. Our projection (~6.11) vs second place (5.975) is a gap of 0.135 = **0.42 SD**; vs V14 (6.049) 0.061 = 0.19 SD. Both are inside test sampling noise.

## What this does not cover

- Image quality uses the 11-06 calibrated estimators (sigma_floor, focus2) and seg_loose legibility only. No new image features were computed.
- The within-donor proxy assumes P1's several frames per donor are comparable acquisitions. The ids do not say whether they are the same eye.
