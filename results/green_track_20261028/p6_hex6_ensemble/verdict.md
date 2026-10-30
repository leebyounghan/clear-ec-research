# p6_hex6_ensemble — verdict (2026-10-28 21:50 UTC, five detectors)

Seed ensembling works where TTA did not. Mean log hex6_all over five fold-clean detectors (stride-8 specialist
s46; stride-2 heads s46, s46-TTA, s47, s48), LOFO blend on HEX:
- base A (single recipe A): -0.124 pp HEX = -0.041 pp mean, 5/5 folds improving, 4/5 folds <= -0.03
  (per-fold mean -0.064 / -0.041 / -0.049 / -0.033 / -0.020). Passes the gate and the pre-registered
  "recommend full-data detectors" threshold on this base.
- base B (hybrid 0.5 recipe A 2-seed + 0.5 Phase-1 five): -0.070 pp HEX = -0.023 pp mean, 5/5 folds improving,
  1/5 folds <= -0.03. Below the parent's 0.03 bar for the submission base; above the 0.02 "absorbed" line.
- base A2 (recipe A two-seed): -0.125 pp HEX = -0.042 pp mean, 4/5 folds <= -0.03.
Single detectors: -0.051..-0.069 pp HEX each (A); three stride-2 seeds -0.100; the detector curve is still
rising at five (2 detectors -0.091, 3 -0.100, 5 -0.124 on A). mean|sides-6| is no better than hex6_all in
the ensemble (-0.112). Eight-detector run (stride-8 s47/s48, stride-2 s49/s50) queued for ~02:00 UTC.
Not raised to the parent before the daily summary: on B it is below the agreed 0.03 bar.

## Addendum 2026-10-29 01:40 UTC — nine detectors (p6b)

Detectors: stride-8 s46/s47/s48, stride-2 s46/s47/s48/s49/s50, stride-2 s46 TTA. Mean log hex6_all:
- base A: -0.145 pp HEX = -0.048 pp mean, 5/5 folds improving, 4/5 folds <= -0.03 (per-fold mean
  -0.073 / -0.046 / -0.059 / -0.039 / -0.023).
- base B (hybrid): -0.085 pp HEX = -0.028 pp mean, 5/5 folds improving, 2/5 folds <= -0.03.
- stride-2 seeds only (5): -0.040 mean on A; stride-8 seeds only (3): -0.028; mixing architectures helps.
Hybrid-base curve by detector count: 2 -> -0.015, 5 -> -0.023, 9 -> -0.028 pp mean. Diminishing: +0.005 per
four extra detectors. The thirteen-detector run (multi-layer stride-8 heads s46/s47, stride-2 s51/s52) is the
last planned point; if it reaches -0.030 on B with HEX improving on >= 4 folds it meets the parent's bar.

## Addendum 2026-10-29 03:45 UTC — thirteen detectors (p6c), saturation

Adding multi-layer stride-8 heads (s46/s47, F1 0.59-0.65) and stride-2 seeds 51/52 (F1 0.60-0.69):
- base A: -0.142 pp HEX = -0.047 pp mean, 5/5 folds, 4/5 folds <= -0.03 (nine detectors: -0.048).
- base B (hybrid): -0.086 pp HEX = -0.0286 pp mean, 5/5 folds improving, 2/5 folds <= -0.03
  (nine detectors: -0.0284). The hybrid-base curve has saturated: 2 -> -0.015, 5 -> -0.023, 9 -> -0.028,
  13 -> -0.029 pp mean. The point-derived HEX signal's ceiling on the submission base is about 0.03 pp,
  i.e. at the parent's bar, not above it; the multi-layer heads are individually weaker (-0.039/-0.045 HEX).
Decision: no more detector seeds. Final numbers go into the daily summary for the parent/PI to weigh
against the deployment cost (each detector is one extra fine-tuned ViT-B pass per image).
