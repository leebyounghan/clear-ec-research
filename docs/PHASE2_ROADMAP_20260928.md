# Phase II research roadmap — September 28

This is a research plan, not a change to the verified September 14 submission.
Confirm invitation and final Phase II rules from the organizer before any
official submission. The public challenge page says the top five advance after
the September 30 review and permits at most two Phase II submissions. The Penn
organizer page currently lists October 1–November 14 for Phase II; treat that
as a planning horizon until an invitation confirms the final deadline.

## Current controlled experiment

Two one-H200 Jobs, `clear-ec-p2-metric35-f2-0928` and
`clear-ec-p2-metric35-f3-0928`, are queued in `p-ajou-mlxp-project`. They do not
preempt the unrelated 8-GPU workloads. Both were Pending when created because
the two eligible nodes had insufficient free GPUs. They expire after seven days
if they never schedule; each scheduled Pod has a 16-hour bound. No more than two
research GPUs are requested concurrently.

The frozen PVC root is `experiments/phase2_metric35_ssl_20260928` under
`/workspace/clear-ec-research`; `results/phase2_metric35_ssl_20260928/manifest.json`
is the local manifest copy. Source is `code/archive/phase2_explore_2026-10/phase2_metric35_ssl.py` and the
Job definition is `k8s/clear-ec-phase2-metric35-ssl-0928.yaml`.

Each fold runs a smoke test, then two fresh fixed-terminal 5,215-update fits.
Both use the same CPSAM metric35 architecture, seed 42, central native256
random training tile, central five-tile D8 inference, slide-averaged training
labels, original validation labels, mixed log/relative loss, batch 48 and
optimizer schedule. The second arm only changes encoder initialization:
6,000-step masked-EMA SSL using that fold's training images (no resized crops,
no validation images or labels). First-three-batch image hashes must match.
The split excludes shared slides and exact-duplicate-linked validation groups.

Primary endpoint: replace only the historical metric35 member in the fixed
six-member development ensemble on the same duplicate-clean rows. The SSL arm
must improve both folds and pooled mean MAPE by at least 0.01 percentage point
to qualify for a third-fold confirmation. Historical OOF use by the other five
members prevents this from being an independent official-score estimate. No
automatic all-data fit, packaging, calibration update, or official submission.

## Decision sequence after the screen

1. **If SSL passes:** confirm on a third clean fold, then evaluate whether the
   marginal ensemble benefit justifies an all-9,000-image fit. If it fails,
   stop trying SSL initialization across more near-identical central seeds.
2. **Address common errors:** use the existing full-frame sixth as a control;
   develop a genuinely complementary cell-morphology/region branch. Prioritize
   CD/CV/HEX extremes, where the current system regresses toward the mean.
   Compare on aligned, duplicate-clean folds with fixed ensemble composition.
3. **Calibration and efficiency:** test simple tail-aware calibration with a
   nested/fresh holdout before adoption. Isotonic calibration previously
   worsened. Distillation is optional after a better teacher exists; it is not
   yet a demonstrated accuracy improvement.
4. **Release:** preserve the verified 8.7010 model as a fallback. Only candidates
   with repeatable fixed-ensemble gains proceed to full-data training, Linux
   container checks, and one of the two Phase II submission opportunities.

No Phase II official submission is authorized by this research plan.
