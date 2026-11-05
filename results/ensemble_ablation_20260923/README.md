# September 23 research ablation (not a submission change)

Recompute with `code/audit_ensemble_20260923.py` and
`code/audit_tail_calibration_20260923.py` on the existing PVC experiment tree.
Full numerical output is in `report.json` and `tail_calibration.json`.

## Paired ensemble ablations

The first five members were compared on 8,954 aligned historical five-fold OOF
images. All scores below are mean MAPE in percentage points; lower is better.

| Equal-weight predictor | Mean MAPE | Difference versus all five |
|---|---:|---:|
| All first five | 9.03552 | — |
| Without seed42 | 9.03189 | -0.00363 |
| Without seed77 | 9.04555 | +0.01003 |
| Without seed2026 | 9.03585 | +0.00033 |
| Without metric35 | 9.05053 | +0.01501 |
| Without predbag2_35 | 9.04573 | +0.01021 |

Residual-error correlations between first-five pairs are 0.955–0.994 across
the three metrics. Three-seed redundancy is plausible; `metric35` and
`predbag2_35` have more consistent marginal value. The apparent gain from
dropping seed42 is smaller than an exploratory image-level bootstrap interval
(-0.01028 to +0.00407) and reverses on fold 2. Do not remove it based on this
screen alone.

The sixth-member matched comparison uses only 5,350 duplicate-clean primary
rows from folds 2–4. Five only: 9.05180; with raw-label sixth: 9.02919; with
standard masked-EMA SSL/raw sixth: 9.01960. Thus the SSL sixth improves the
fixed six by 0.00959 versus the raw sixth and adds 0.03220 versus five only.
These are historical internal estimates, not the submitted all-data model's
leaderboard score.

## Common-mode error

The five-model mean regresses toward the middle. In the bottom target decile,
signed bias is +12.96% CD, +19.63% CV and +22.64% HEX; in the top decile it is
-8.94%, -17.09% and -15.58%. Middle-decile MAPE is substantially lower than
tail MAPE. This may reflect loss/label uncertainty and/or missing morphology,
not simply weak pretraining. Diagnostic tile analysis already showed that a
ground-truth oracle crop was much better than feasible learned/quality-only
crop aggregation; the oracle cannot be deployed.

Exploratory fold-cross-fitted power-law calibration improved the five-member
score from 9.03552 uncalibrated and 9.00459 cross-fitted scalar to 8.98374;
it improved each of five folds. Isotonic calibration worsened to 9.04060.
This is a hypothesis to test in a fully nested or fresh holdout design, **not**
a validated replacement for the official fixed calibration. The meta-layer
uses historical base OOF predictions whose base-model training is not nested
relative to the calibration outer folds.

## Next GPU experiment, when capacity becomes available

1. Use at most four H200 GPUs: two fold-clean splits × matched raw-initialized
   versus standard masked-EMA-initialized **metric35** models, holding crop,
   original labels, objective, update count, and random sampling fixed.
   Predeclare evaluation on the same primary images and the fixed remaining
   ensemble; exclude each fold's validation images and duplicate-linked slides
   from SSL. Use terminal checkpoints, not validation-selected updates.
2. Keep SSL only if it improves the fixed ensemble on both folds and the pooled
   mean by at least 0.01 point. Then confirm on a third fold. The prior 6,000-step
   SSL itself took roughly 2 H200-hours per fold; full downstream training adds
   several hours per run. Re-estimate from live progress after scheduling.
3. If this fails, stop adding near-duplicate CPSAM seeds. Prioritize a
   weakly supervised morphology/region branch and a genuinely nested
   tail-calibration test, retaining metric35, predbag2_35 and the useful
   full-frame sixth as controls.

No GPU experiment was launched on September 23: a one-GPU scheduling probe
could not schedule because both eligible H200 nodes reported insufficient GPU;
the two unrelated 8-GPU workloads were left untouched. No official checkpoint,
Docker image, or shared submission file was altered.
