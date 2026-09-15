> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../TARGET_REPLICATION_20260912.md) for the release-facing summary.

# Target-policy replication: launched September 12

## Verified state at 15:22 UTC

- `clear-ec-replication-0912`, pod `clear-ec-replication-0912-sf2pv`: Running on `h200-04-w-11f1`, four H200 GPUs, all four full-training workers active after four successful smokes. GPU utilization 98–100%, memory 51,769 MiB per device. First 25 updates took 74.8–75.1 seconds, approximately 3 seconds/update.
- Fold2 and fold3 each compare fresh original-CPSAM initialization, seed77, raw versus slide-averaged targets. Fixed 35 epochs / 5,215 updates per worker. Same native256 full_quality K2 views, batch48, mixed MAPE, encoderLR1e-5/headLR1e-3, BF16 and fixed40-view evaluation. No additional SSL pretraining or tuning on validation.
- Primary split sizes: fold2 train7153 / validation1791 / historical-duplicate-clean primary1781; fold3 train7161 / validation1791 / primary1791. Removed10 training rows from fold2 and zero from fold3. Slide and exact-duplicate-connected train/validation overlap zero. All8954 raw image hashes checked. Synthetic tests5/5 passed on Kubernetes, not the Mac.
- Parent screen finished successfully at05:29:53UTC. Replication was designed earlier but was NOT actually launched overnight; it was launched after parent results were read. Do not describe this as prospectively registered.

## Additional capacity and automatic continuation

`clear-ec-fold4-0912` requests two additional H200 for matched raw/smoothed fold4 fits. CPU preflight passed: train7150, validation1790, primary1778,12 duplicate-connected training rows removed. Tests5/5 passed. Each fit has5,180 updates.

At15:22UTC its pod was Pending with no assigned node: the scheduler reported insufficient GPUs on two nodes and insufficient CPU on one; other nodes did not match the permitted zone/selector. Other project workloads remain untouched. This is a real Kubernetes queued training Job, not an app-only reminder. It starts when two GPUs and the requested CPU/memory become available, including after the four-GPU batch exits. Pending allocates no GPUs.

The four-GPU Job has a7h wall deadline including scheduling; the two-GPU Job allows12h including waiting. Each container is separately capped at6h runtime, worker5.5h, no retries, fail-stop, automatic audit and exit,10-minute finished Job TTL. Maximum container GPU allocation is approximately24+12=36GPU-hours (plus at most30-second kill grace), not an unlimited queue. Actual expected train/evaluation compute is approximately18GPUh plus9GPUh. At the observed throughput, first batch results are expected around20:00UTC September12, with the queued fold4 batch approximately4.5h after it starts. Scheduling and throughput may change these estimates.

## Artifacts and interpretation

- Main PVC folder: `experiments/overnight_target_20260912/`.
- Extra fold4 folder: `experiments/target_fold4_20260912/`.
- Each has frozen `source/`, a hash manifest, private split/reference arrays, status/progress logs, and eventual `screen_audit.json` plus `queue_finished.json`. Never infer success merely from pod disappearance. Failure markers include `complete.json.failed`, `interrupted.json`, and `audit_failed.json`.
- Main frozen source predates the local generic fold-selection extension. This is intentional; do not overwrite it. The fold4 source uses explicit `CLEAR_EC_REPLICATION_ROOT` and `CLEAR_EC_REPLICATION_FOLDS=4`. Local source edits do not alter already frozen main training.
- Compare raw against NEW matched smoothed seed77 controls, standalone and fixed equal-weight six-model replacement. The historical sixth on these folds is seed42, NOT the submitted regional seed77 sixth. These are reused internal folds, not a leaderboard forecast or untouched test set. Single fold4 alone cannot pass a two-fold replication gate.
- No changes to the September10 candidate, no automatic model promotion, packaging, challenge submission, external data upload, or GitHub push.

The first preflight Job produced no pod because its explicit nonpreemption override conflicted with the cluster's fixed private-zone priority policy. That empty Job was replaced using the cluster-required policy; all known private-zone project workloads have equal priority. The actual extra-job scheduling event confirmed that preemption would not help. No existing workload was stopped or modified.
