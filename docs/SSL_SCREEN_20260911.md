# Native-scale, fold-clean SSL screen — 2026-09-11

## Completed — September11 22:49UTC / September12 07:49KST

All4 full workers succeeded, strict reload and audit passed. Queue elapsed
25,970.28s (~7h13m); audit finished22:49:32UTC. Primary3568-row standalone
means: clean35 9.261761, SSL35 9.208103, continued55 9.265722. Current6
8.975492 -> SSL fixed replace6 8.971051, delta -0.004441. Both folds improved
(-0.003046/-0.005831), but the predeclared0.02 magnitude gate FAILED.
`passes_screen=false`, no confirmation, promotion, packaging or submission.

Fixed training-only teacher probes did not show prior severe contraction:
token std0.543->0.516 and0.482->0.542; effective ranks increased. These probes
do not prove downstream usefulness or absence of all representation failures.
Measured training/pretraining/evaluation total27.36GPUh (do not double-count
the clean35 stages included in continued55). Queue allocation~28.86GPUh,
excluding startup/audit. These are not billing amounts.

At the later23:33UTC check the Job and GPU pod were absent, consistent with
configured completion TTL; queue markers confirm successful completion.
Read-only CPU report recovery completed; its pod was deleted. Original500GiB
PVC remained Bound. Twelve local report/probe files in
`results/ssl_screen_20260911/` matched remote SHA256s. No image/label/OOF arrays
were copied. The next target-policy experiment is a separate root; never restart
this completed queue. The historical launch details below are not current status.

## Live launch verified — 15:38 UTC / September12 00:38 KST

- Job `clear-ec-ssl-screen-0911`, pod `clear-ec-ssl-screen-0911-fs8mv`,
  node `h200-04-w-11f1`, four allocated H200.
- Synthetic CPU tests:3 passed. Four actual GPU smoke workers passed strict
  checkpoint reload and40-view inference on4 samples each. The first2 downstream
  minibatches match hash-identically between SSL and clean controls in both folds.
- All four FULL workers launched after the smoke barrier at ~15:37 UTC;
  no full worker failed at this launch check. No scores are available yet.
- Snapshot GPU utilization98%,99%,100%,100%; process memory20,329MiB on each
  SSL worker and51,769MiB on each supervised worker. This is a snapshot, not
  a sustained-utilization guarantee or evidence of performance improvement.
- First25 full SSL updates:1.217s/update and1.212s/update. At this early rate,
  6000 updates take~2.0h before probe/checkpoint overhead. Downstream~4.4-4.6h
  plus evaluation; controls35+20~7h plus two evaluations. Provisionally expect
  the complete screen~7-9h after launch, roughly September12 morning KST.
- Hard Job/pod deadline12h from creation15:36UTC. Controller stops after11h
  and limits each worker to10h. No retries. Kubernetes native Istio sidecar
  verified (`restartPolicy: Always` in initContainers), so main completion exits
  the pod. TTL deletes the finished Job/pod after600s; PVC results persist.
- Fold0 removes10 duplicate-linked training rows, leaving7151; fold1 removes4,
  leaving7157. Primary validation1781+1787=3568; all1791+1791 historical validation
  rows are still evaluated for a separately labelled descriptive report.
- Local launch provenance is under `results/ssl_screen_20260911/`.

The following sections describe the fixed experiment, not completed results.

User authorized implementation and then removed the earlier two-GPU limit.
Request four H200 for four independent useful workers; do not preempt or modify
other workloads. Node inventory is forbidden; four is an experimental request,
not a verified claim about the maximum free hardware in the cluster.

## Frozen question and scope

Does additional CLEAR-EC TRAIN-ONLY native-scale masked feature pretraining with
explicit variance/covariance regularization improve the existing six-member
ensemble? This is a JEPA-inspired custom ablation, NOT official I-JEPA, VICReg,
or LeJEPA. No new architecture, external data, downloaded weights, crop-policy
changes, calibration fitting, deployment, confirmation or submission is included.
Keep the September 10 package immutable. CLEAR-EC data stay on its private PVC.

Initial CPU preflight found exact full-image duplicate leakage in the historical
splits: fold0 has6 train/6 validation images across6 hashes, fold1 has2/2 across2.
8954 input images contain8940 unique pixel arrays. None of the affected training
images is constant. The historical control is therefore NOT reused for scoring.
Preserve that failed preflight; before any GPU result, revise the experiment to
exclude whole training slide groups connected to validation via exact duplicates
(including transitive connections), recompute training-only smoothed labels, and
train NEW clean35 controls. For comparison with the historical six-member ensemble,
primary validation also excludes its duplicate-linked validation slide groups.
Full historical validation metrics are descriptive only.
Four new full workers:

- Fold0: fresh original CPSAM -> 6,000 SSL updates -> fresh regression heads,
  identical terminal35-epoch downstream fit.
- Fold1: same, independently pretrained WITHOUT fold1 validation images.
- Fold0: fresh original CPSAM -> terminal35-epoch CLEAN supervised fit, preserve
  checkpoint/predictions ->20 more epochs at encoder LR3e-6/head3e-4 with fresh
  optimizer/cosine and seed177 -> continued55 control.
- Fold1: same clean35 plus continued55 supervised control.

The continued55 control is an approximate extra-compute comparison, NOT exact
wall-time/FLOP matching or a single uninterrupted55-epoch cosine schedule.
Report measured GPU-hours and quality for all branches. Two control checkpoints
per fold are predeclared comparators, never selected by validation performance.
Original35 fits used ~4.3h training per fold;55 scales to ~6.8h. Prior similar
SSL6,000 took ~2h, so SSL+35 is provisionally ~6.5-7.5h with evaluation and I/O.
Budget 8-10h after startup, hard12h pod deadline (at most48 requested GPU-hours).
New SSL regularization and contention can change throughput; update using runs.

## Data and method isolation

Each SSL worker samples ONLY its own historical supervised training indices.
Image IDs and slide groups are disjoint from that fold's validation set; exact
full-image duplicate overlap is checked before launch. Available grouping is
slide_id, not verified patient identity. Upstream original CPSAM data are not
newly audited. Near-duplicate and unrecorded patient overlap remain limitations.
Do not reuse either fold's SSL initializer for confirmation on other folds.

Use original full_quality per-image native256 selection with jitter, D8 and
weak brightness/contrast only; no image resize or cross-region invariance loss.
Online masked context predicts an EMA teacher's SAME-crop target feature map.
Add variance floor and off-diagonal covariance penalties to normalized ONLINE
encoder tokens, not just the predictor. Fixed training-only probe checks token
variance and effective rank at initialization and every500 updates. Fail closed
on nonfinite features or severe relative deterioration; loss is not a score.
6000 is fixed in advance; no selecting checkpoints with downstream validation.
After SSL, restore seed77 before building fresh heads; supervised batch sampling
must hash-identically match the NEW clean35 control in the first3 updates.

## Predeclared decision (before any new results)

Primary ensemble: REPLACE the sixth full_quality_seed77 member at weight1/6:
new6 = (5 * old5 + candidate) / 6. No tuned weights or calibration.
Apply the SAME replacement to new clean35 control and continued55 control.
Screen-positive requires SSL replacement:

1. improves current6 in BOTH folds and >=0.02 percentage points in aggregate;
2. beats new clean35 replacement in BOTH folds;
3. beats continued55 replacement in BOTH folds.

All gates use the predeclared duplicate-connected-group-clean primary validation
subset. No new model scores had been seen when this integrity correction was made.

Report standalone metrics, CD/CV/HEX deltas, fold consistency, paired per-image
error wins/losses, correction versus damage magnitude, and measured compute.
Add7 may be a descriptive diagnostic only; it cannot rescue a failed gate.
All historical folds have been reused in development: no significance claim or
leaderboard forecast. Passing is eligibility for review, not automatic promotion.

## Execution safeguards

CPU preflight first; verify original weights, saved control artifacts, sources,
split identities, fullframe manifest and current6 reconstruction. Four GPU smoke
tests must pass before any full worker starts. Strict checkpoint reload, finite
positive outputs, no automatic retries, finite controller and pod deadlines.
Any failure stops remaining workers. Controller writes aggregate audit and exits
after four jobs; the original data, weights and PVC are never deleted.
