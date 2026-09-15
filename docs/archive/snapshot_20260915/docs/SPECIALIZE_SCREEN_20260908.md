> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../SPECIALIZE_SCREEN_20260908.md) for the release-facing summary.

# Two-GPU matched specialist screen — September 8, completed

## Final result — 18:33 UTC audit

Queue completed at18:27:05UTC with folds0/1 done, failed={}, active={}.
The one-time audit completed at18:33:33UTC: NO specialist passed the
predeclared screen. No remaining-fold queue was launched and no candidate
was promoted, packaged, uploaded or submitted.

The comparison below uses exactly the SAME3582validation images from folds0/1.
Current6 overall is8.975950250 on this subset, NOT its full5fold score and NOT
a leaderboard prediction. Positive deltas mean worse MAPE (percentage points).

| Specialist | Target vs matched multitask | Overall with fixed50/50target blend | Overall vs current6 | Blend vs matched-control blend | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| CD | +0.041613 | 9.030822 | +0.054871 | +0.005911 | No |
| CV | +0.007642 | 8.995410 | +0.019459 | +0.001460 | No |
| HEX | -0.038487 | 8.987476 | +0.011526 | -0.006162 | No |

HEX improves its matched-control target in both folds, but the aggregate
improvement0.038487 is below0.05 and its blend worsens current6 in BOTH folds.
CD/CV do not show an aggregate specialization benefit. These results reject
THIS20+5epoch-equivalent specialization recipe; they do not prove that all
independent-encoder models or longer specialization training are ineffective.
No fitted blend-weight search was added after observing these results.

Next decision: retain existing candidates; do not spend three more folds on
this failed recipe. A different representation/pooling experiment would need
a separately bounded plan and matched control, not an automatic continuation.
Submission packaging still needs resolution; no localDocker repair was done.

Audit: `SPECIALIZE_SCREEN_20260908_audit.json` (local copy of remote report).
SHA256:`cb174ba673ac6d3911b866dc5a54456dfc5a606028410b750678fecf003502aa`.
ID/target/split alignment passed;6frozen source hashes and all10full-run
checkpoints were verified against recorded hashes. Eight full1791image
OOF outputs and both fold completion records exist. Historically reused OOF
remains exploratory, not an untouched test.

Measured common+branch training9.930542GPUh, evaluation1.624671GPUh,
total11.555214GPUh. Queue12:38:47–18:27:05UTC =5h48m18s wall on2H200,
11.61allocatedGPUh during the queue (excluding pod startup and post-run audit).
Each full-run checkpoint is~1.219GB; experiment directory uses23GiB including
smokes. All remain under `/workspace/clear-ec-research/experiments/specialize_screen_20260908`
on the existing mountedPVC `clear-ec-workspace`, verified as Lustre volume
`pvc-exa-abfcdbc5-269d-4f5c-a069-6716e1606c6a`. No model/data files were deleted.
At18:36UTC both GPUs0%/0MiB with no compute processes; all workers/controller
and CPU audit had exited. Graceful deletion of ONLY`clear-ec-specialize-2gpu`
was requested. Deletion wait completed successfully; at18:37UTC a subsequent
read confirmed the pod absent and PVC stillBound on the same volume. TwoH200
were released; old8GPU pod was not recreated. Heartbeat`clear-ec-2gpu` was
deleted through the app after completion. The brief terminating-pod Error
status during graceful deletion was after successful experiment completion,
not a training failure. Model/data artifacts remain recoverable via the PVC.

## Historical launch record (not current status)

User approved shared-training -> per-metric specialization, using only2or4 GPUs.
New pod `clear-ec-specialize-2gpu` in `p-ajou-mlxp-project` requests exactly2H200,
24CPU and128Gi memory (400Gi limit). It mounts existing `clear-ec-workspace` PVC;
original8GPU pod remains deleted. No existing staged models/data were changed.
Manifest: `k8s/clear-ec-specialize-2gpu.yaml` (pod only, no PVC creation/deletion).

Latest12:43:37UTC: both workers healthy, failed={}, latest progress50/2980
at12:42:45–46UTC,3.240–3.245s/update, both GPUs100%,51769MiB. Ten historical
baseline member paths and the audit script import were also verified.
Heartbeat `clear-ec-2gpu` checks every10minutes, quiet for normal progress,
audits completed results and releases this idle pod while retaining the PVC.

Remote repo `/workspace/clear-ec-research`; Python `/workspace/venv/bin/python`.
Experiment `experiments/specialize_screen_20260908`; controllerPID255,
fold0worker395/GPU0 and fold1worker398/GPU1. Both initial smoke workers323/326
completed before full launches around12:40UTC. At12:41:27UTC both full workers
were at25/2980 common updates, failed={}, both GPUs100%,51769MiB each.
PyTorch2.11.0+cu130 and two visible GPUs verified. Source is frozen under this
new experiment's `source/`, with hashes in manifest.json. Do not edit running
source or rerun the queue. Audit script lives separately at experiment root.

## Controlled comparison

- Two existing slide-group folds0/1, each1791validation images;8954total.
- Exact historical split/ID/target agreement checked, positive training labels,
  disjoint train/validation slide sets, smoothing uses TRAIN frames only.
- Fresh original CPSAM initialization for each fold. NO existing all-data or
  validation-selected checkpoint is used as the common initialization.
- Full-quality native256 crop policy, K2, BS48, mean pooling, separate3heads,
  mixedMAPE, fixed legacy LM/LS, BF16, activation checkpointing, clipping1.
- Fixed common2980updates (20*149), encoderLR1e-5/headLR1e-3, cosine/warmup.
- Four branches each restore the SAME terminal common checkpoint, then get
 745updates (5*149), fresh AdamW, encoderLR3e-6/headLR3e-4 and the same cosine
 schedule. Branches: multitask control, CD-only, CV-only, HEX-only.
- Specialists update encoder+normalization+their own head. Other heads freeze;
  all heads are still computed to preserve dropout RNG consumption. Only the
  specialist's target output is valid for adoption; ignore its other metrics.
- Identical branch RNG seeds and sample order. Initial crop tensor/index hashes
  must match between branches. Smoke tested one-target loss isolation,
  optimizer coverage, frozen-head gradients, save/load, all4branches and40-view
  finite positive prediction on4images/fold. Smoke scores are NOT performance.
- Validation occurs only AFTER each terminal branch checkpoint is saved, on
  all1791images×40views, encoded-view mean then exp decode. No best-epoch search.
  Historically reused folds remain exploratory, not independent test data.

Each resulting branch has25epochs-equivalent total optimization. This is NOT a
matched-budget comparison to the previous35epoch regional candidates; it is a
matched comparison of specialization against continued multitask learning.
Only results against both that matched control AND the current6ensemble matter.

## Fixed audit and decision gate

`audit_specialize_screen.py` reads completed outputs and historical5+full77
six-model OOF, verifies alignment and records input SHA256. For each specialist,
blend only its own output50/50with current6; other two current6outputs unchanged.
Also report the corresponding50/50multitask-control blend. No fitted weights,
calibration or greedy combination is used for screening.

Eligibility requires target MAPE improvement over matched control in BOTH folds
and aggregate>=0.05points; fixed specialist blend must improve overall MAPE in
BOTH folds and aggregate>=0.01points vs current6. Eligibility means remaining
fold confirmation is warranted, NOT automatic candidate promotion. Report
whether specialist blending actually beats matched-control blending as well.
If no target passes, do not extend this queue or claim8.x performance.

After both fold workers finish, run once from remote repo root:

```
/workspace/venv/bin/python experiments/specialize_screen_20260908/audit_specialize_screen.py
```

Outputs: common/selected.pt, each branch selected.pt/oof.npz/result.json,
fold split/config/result.json, queue status.json/complete.json, screen_audit.json.
Existing5and6candidates, frozen prior source and9000raw images remain unchanged.

## Resource/time budget

Two fold workers execute concurrently; each owns one GPU and serializes branches.
Total full training11920updates =2*(2980+4*745). At first25updates measured
3.250–3.263s/update, ~10.8GPUh for training (~5.4h wall on2GPUs), plus8full
1791image40-view evaluations and checkpoint I/O. Planning estimate6–7h wall,
around18:40–19:40UTC Sep8 (14:40–15:40NewYork /03:40–04:40KST Sep9).
This is provisional and must be updated using longer-run progress rather than
the two-update smokes. No cloud price is known; no currency-cost claim.

Controller has30min smoke/9h full-worker bounds,18GPUh full upper allocation,
absolute stopSep8 23:30UTC, no retries. Only its own timed-out process groups
can be terminated. It does not delete pod/PVC or spawn further experiments.
Monitor completion/failure and keep normal progress quiet. After terminal jobs,
finish CPU audit and verify persistent artifacts/no active work, then release
only this idle pod (per user's idle-GPU instruction), retaining PVC. Never
recreate8GPU allocation or repair localDocker as part of this screen.
Remaining-fold confirmation needs a bounded follow-up plan consistent with
Sep10/14submission preparation; do not silently expand the experiment queue.
