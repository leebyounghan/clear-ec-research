# G3 — hex6-auxiliary multi-task fine-tune of the recipe A regressor (2026-11-01 21:00 – 2026-11-02 00:00 UTC)

Pre-registered 21:10 UTC (protocol.json). Ten runs: recipe A seed-46 fold checkpoints fine-tuned 2 epochs (894 updates,
batch 16, encoder lr 3e-6, head 3e-4, ~50 min each on one H200) with an auxiliary linear head on the 128-d hidden layer
predicting the fold-clean active ten-family mean-log hex6 (lambda 1), against an identical lambda-0 control. TTA4
predictions on the held fold; reconstruction of A and B from the two original members is exact (max rel diff 0).

## G3c (decision-relevant): distilled hex6 cannot replace the detectors — FAIL

The aux head's hex6 prediction correlates 0.855 (log) with the active detector feature, but LOFO-blended alone into B's
HEX it gets w = 0 / 0 / 0 / 0.02 / 0.02: **+0.0001 mean pp** (HEX +0.0002), 0/5 folds improving (gate <= -0.02).
Reading: the hex6 gain comes from information that is not in the regressor's pooled representation (independent error);
a head trained on that representation can only reproduce the part the regressor already uses. The ten detector passes
stay necessary for the -0.031 pp.

## G3a (representation) — FAIL

Single-seed s46 TTA4 x LOFO scalars, HEX MAPE:

| arm | f0 | f1 | f2 | f3 | f4 | folds 0-3 pooled | fold 4 |
|---|---|---|---|---|---|---|---|
| baseline (untouched epoch 20) | 10.580 | 10.025 | 10.289 | 10.217 | 10.154 | 10.2778 | 10.1542 |
| aux (lambda 1) | 10.600 | 9.994 | 10.280 | 10.237 | 10.128 | 10.2776 | 10.1276 |
| control (lambda 0) | 10.562 | 10.094 | 10.301 | 10.246 | 10.140 | 10.3007 | 10.1401 |

aux - control = -0.0231 pp on folds 0-3 (gate -0.10), fold 4 -0.0125. The fine-tune itself is neutral to slightly
harmful (+0.023 vs baseline) and the auxiliary objective only brings it back to baseline. Mean score per fold moves
within +-0.04 either way.

## G3b (submission) — FAIL

B with its s46 member replaced by the arm (A = mean(arm, s47) x re-fit LOFO scalars, B = 0.5 A + 0.5 P1), then the
active hex6 blend re-fit on top (8934 non-leaked rows, B = 8.8682 there, B + hex6 = 8.8370):

| arm | B_new | hex6 on B_new (mean pp) | combined | vs B + hex6 | folds improving |
|---|---|---|---|---|---|
| aux | 8.8621 | -0.0270 | 8.8351 | -0.0019 | 3 |
| control | 8.8667 | -0.0315 | 8.8351 | -0.0019 | 3 |

Identical combined result for both arms: whatever the fine-tune changes, the hex6 blend absorbs it; the auxiliary
objective contributes nothing (gate -0.03).

## Status

G3 closed. G3d (lambda 3, same schedule, pre-registered 23:55 UTC) runs as the dose-response confirmation on GPUs 2-3
(folds 0-3) and GPU 0 (fold 4); its gate is the same as G3a against the control arm.

Files: `g3_hex6aux/{result.json, result_partial.json}`, per-run `g3_hex6aux/<arm>/fold<k>/{result.json, predictions_tta4.npz, model.pt}`,
`source/green_day2_g3_hex6aux_20261101.py`, `source/green_day2_g3_evaluate_20261101.py`.

## G3d — lambda 3 dose-response (2026-11-02 01:40 UTC): confirms the null, in the wrong direction

| arm | HEX f0 | f1 | f2 | f3 | f4 | folds 0-3 pooled | fold 4 | B-level combined vs B + hex6 | distilled hex6 on B |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 10.580 | 10.025 | 10.289 | 10.217 | 10.154 | 10.2778 | 10.1542 | | |
| lambda 1 (aux) | 10.600 | 9.994 | 10.280 | 10.237 | 10.128 | 10.2776 | 10.1276 | -0.0019 (3/5) | +0.0001, w 0/0/0/0.02/0.02, r 0.855 |
| lambda 3 (aux3) | 10.688 | 10.015 | 10.324 | 10.290 | 10.161 | 10.3292 | 10.1610 | +0.0080 (0/5) | +0.0002, w 0.02/0/0/0.04/0.02, r 0.853 |
| control (lambda 0) | 10.562 | 10.094 | 10.301 | 10.246 | 10.140 | 10.3007 | 10.1401 | -0.0019 (3/5) | |

A three-times stronger auxiliary objective makes HEX worse than the control (+0.0286 pp on folds 0-3, +0.021 on fold 4),
costs CD on some folds (fold 3 raw CD 7.08), and does not make the distilled hex6 any more informative (r 0.853 vs 0.855,
blend weight still ~0). The hex6 information cannot be pushed into the regressor's representation by this route.
G3 closed on all gates. Files: `g3_hex6aux/result_aux3.json`, `g3_hex6aux/aux3/fold<k>/`.
