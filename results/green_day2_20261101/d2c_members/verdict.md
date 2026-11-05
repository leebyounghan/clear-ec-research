# D2c / D2d — new full-frame members and tissue filtering in the ACTIVE hex6 (2026-11-01, 21:05-21:18 UTC)

Control = plain ten-family LOFO on B: -0.0311 mean pp (HEX -0.0934), reproduced in every run.

## D2c — frame-augmented pre-only detector (aug_s46, TTA4) as a member: gate FAILED

| composition | mean pp | HEX pp | folds improving | per fold (mean pp) |
|---|---|---|---|---|
| control ten | -0.0311 | -0.0934 | 5 | -0.0473 -0.0309 -0.0415 -0.0274 -0.0086 |
| swap sg2pre_s46 -> aug_s46 | -0.0293 | -0.0879 | 5 | -0.0460 -0.0309 -0.0367 -0.0275 -0.0056 |
| ten + aug_s46 | -0.0315 | -0.0944 | 5 | -0.0499 -0.0346 -0.0379 -0.0299 -0.0052 |
| single aug_s46 | -0.0048 | -0.0145 | 3 | |
| single sg2pre_s46 (shipped member) | -0.0107 | -0.0320 | 5 | |

Adding the augmented detector gains 0.0004; swapping it in loses 0.0018. The frame-augmented detector finds more cells
(840-1030 per image vs 660) but its hex6 is a weaker HEX signal under the deployed definition.

## D2d — deployed-definition hex6 on tissue-filtered peaks: no effect

Filtering peaks to the non-black tissue mask changes the deployed hex6 on 0.5% of rows (mean abs diff 0.0000) for both
full-frame members; every composition is identical to its unfiltered counterpart (ten with filtered pre -0.0311; ten +
filtered aug -0.0315). The single-member discrepancy seen earlier (aug_s46 -0.0113 under the D1 feature definition vs
-0.0048 under the deployed one) is a hex6-definition difference (bounded-cell / six-vertex counting), not a tissue
effect; the deployed definition is the better one for the shipped member (-0.0107 vs -0.0089), so it stays.

TTA members (s46/s47/s48 pre-only with four-flip TTA) are evaluated in the same way when G1 finishes (result_tta.json).
Files: `d2c_members/result_aug.json`, `d2d_tissue_filter/result.json`, `source/green_day2_d2c_members_20261101.py`, `source/green_day2_d2d_tissue_filter_20261101.py`.

## D2c — TTA pre-only members (G1): gate FAILED (2026-11-01 23:58 UTC)

| composition | mean pp | HEX pp | folds improving |
|---|---|---|---|
| control ten | -0.0311 | -0.0934 | 5 |
| ten + tta_s46 / + tta_s47 / + tta_s48 | -0.0317 / -0.0322 / -0.0325 | -0.095 / -0.097 / -0.097 | 5 |
| ten + all three TTA members | -0.0337 | -0.1011 | 5 (gain -0.0026) |
| swap sg2pre_s46 -> tta_s46 / s47 / s48 | -0.0299 / -0.0301 / -0.0296 | | 5 |
| nine + all three TTA (pre swapped out) | -0.0322 | -0.0967 | 5 |
| single tta_s46 / s47 / s48 | -0.0082 / -0.0119 / -0.0099 | | 4 / 4 / 5 |
| single sg2pre_s46 (shipped, no TTA) | -0.0107 | -0.0320 | 5 |

Four-flip TTA does not improve a pre-only member (s46 gets worse, s47 slightly better); adding three TTA members gains
0.0026, below the 0.005 gate and far below the 0.02 adoption bar. CD from TTA peaks: w = 0 everywhere (D1 definition,
`d1_cd_features/result_tta.json`). G1 closed, evidence-only.

## D2c — seeds 49/50 and the member-count curve (2026-11-02 02:20 UTC): saturated, gate FAILED

| composition | mean pp | HEX pp | per fold (mean pp) |
|---|---|---|---|
| control ten (nine fine-tuned + pre-only s46) | -0.0311 | -0.0934 | -0.0473 -0.0309 -0.0415 -0.0274 -0.0086 |
| ten + tta_s49 (best single new member) | -0.0334 | -0.1001 | -0.0497 -0.0348 -0.0397 -0.0303 -0.0125 |
| ten + tta_s50 | -0.0324 | -0.0973 | |
| ten + three TTA members (s46-48) | -0.0337 | -0.1011 | |
| ten + five TTA members (s46-50) | -0.0334 | -0.1001 | -0.0516 -0.0411 -0.0333 -0.0332 -0.0077 |
| nine + five TTA members (shipped pre-only swapped out) | -0.0337 | -0.1012 | -0.0521 -0.0407 -0.0330 -0.0324 -0.0105 |
| singles: tta_s46 / s47 / s48 / s49 / s50 | -0.0082 / -0.0119 / -0.0099 / -0.0123 / -0.0114 | | shipped pre-only s46 (no TTA): -0.0107 |

Independent full-frame members saturate at about -0.034 mean pp from three members on (three -0.0337, five -0.0334);
the best gain over the shipped ten is -0.0026, below the 0.005 gate and far below the 0.02 adoption bar. The
multiplicity caveat of the earlier detector-count curve stands: more members buy almost nothing once one independent
(never fine-tuned) member is present.
