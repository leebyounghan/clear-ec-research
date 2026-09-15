# Standard SSL duration comparison — September 14, 2026

User requested continued SSL optimization while the independent all-data release
finishes. Release jobs, artifacts, calibration and delivery heartbeat stay untouched.

## Evidence and hypothesis

The completed same-CPSAM DINO/I-JEPA wave did not beat standard SSL on the same
3572 rows: standard9.292635, DINO9.335936, I-JEPA9.319192. Thus no duplicate DINO
wave or official-backbone probe recovery is being launched here. The earlier
ViT-B/8 probe failure remains a separate unresolved evaluation issue.

This wave tests saved standard-SSL teacher checkpoints2000/4000 versus6000.
These are EARLY checkpoints from a6000-step cosine schedule, not independent
shorter cosine schedules. Hypothesis: useful adaptation could peak before6000.
It is not assumed that an earlier checkpoint is better.

Use existing fold-clean teachers from raw_ssl_next_20260913/ssl_runs/fold{2,3}/pretrain.
SSL validation images used=0. The user's all9000 submission SSL continues separately.
This matched experiment isolates checkpoint age instead of changing data exposure.

## Fixed experiment and checks

Each fold gets one H200 and sequentially fits2000 then4000 initialization.
Four fresh regression fits total, each with a two-update/four-image smoke first.
Same raw targets, seed77, full_quality native256, K2/batch48, encoder1e-5/head1e-3,
5215 updates,40-view evaluation. Freeze and hash all inputs/source/teachers/controls.
Check strict checkpoint reload and first-three-batch hashes against standard control.
Audit exact split/IDs/targets; keep valid targets separately per metric, including
the two invalid HEX rows' still-valid CD/CV entries. CPU metric boundary tests
cover zero, NaN, empty metric, and ordinary-positive parity.

Primary endpoint: single-model mean of CD/CV/HEX MAPE, same3572 primary rows.
Choose2000 versus4000 on fold2 only. Research gate requires pooled mean at least
0.03 below6000 and improvement over both6000 and raw in each fold.
Report all methods/folds regardless. No automatic promotion, all-data retraining,
ensemble fitting, packaging, official submission or Git push. Reused development
folds are not independent validation.

## Execution

Namespace p-ajou-mlxp-project; PVC clear-ec-workspace.
Remote root /workspace/clear-ec-research/experiments/ssl_duration_20260914.
Frozen source source/ssl_duration_20260914.py, manifest.json.
Jobs clear-ec-ssl-duration-f2-0914 and clear-ec-ssl-duration-f3-0914.
One H200/6CPU/24Gi request64Gi limit each; maximum two research GPUs.
Existing release uses one separate GPU, so CLEAR-EC total is three while both run.

Previous matched downstream timing2.97s/update:
5215*2.97=15488.55s (4.30h) + approximately689s evaluation =4.49h per fit.
Two serial fits/fold imply about9h wall time and18GPUh for four fits, plus checks.
These are estimates, not a deadline guarantee. Queue cap11h per fold,
22 research-runtime GPUh; scheduling/termination can add overhead.
Outer command39900s, Job41400s, no retry, completed Job TTL600.
Each fold releases its GPU immediately after both fits; no idle GPU waiting for audit.

State: queue_fold{2,3}/{status,complete,failure}.json and per-step smoke/full logs.
Results: runs/step{2000,4000}_fold{2,3}/{config,result,complete}.json and progress.
Do not recreate a missing Job: inspect PVC first, since TTL removes completed jobs.
No automatic retry on failure. Existing other fold can finish within its bound.

After BOTH queues complete, run on a bounded GPU-free helper:
  /workspace/venv/bin/python experiments/ssl_duration_20260914/source/ssl_duration_20260914.py audit
Only if comparison.json does not exist. Verify the aggregate and copy manifest/
comparison/queue completion JSON to local results/ssl_duration_20260914.
All images, labels, OOF predictions and checkpoints remain on PVC.
Clean up only the task-created helper. Report the final table. The app permits
only one heartbeat per thread, so clear-ec-ssl now covers BOTH delivery and this
research, with the full prior delivery scope preserved. Report submission readiness
immediately without waiting for research. Pause only after both branches finish or
fail; a completed research branch must not stop pending submission delivery.
Absolute research follow-up deadline September15 08:00UTC. No new GPU waves.

## Launch status

CPU preflight passed, including metric boundary tests and frozen teacher/control
hash checks. Manifest SHA256:
22441378f320505ee4b6bcc97fcce1a12900315767bede06239966c4e6d8f353.
Source SHA256:
e1451d30e2a01c43afe2a73f2e147729a3c78c8a0f2eeafb70dcac4b91c9a910.

Both Jobs created/started September14 18:45:01UTC on h200-04-w-11f1.
Pods clear-ec-ssl-duration-f2-0914-7d49c and
clear-ec-ssl-duration-f3-0914-nsg84. Both first two-update smoke fits and strict
reload/four-image evaluation passed at18:45:55–56UTC. The full step2000 fits
were dispatched18:45:56–57UTC. Fold2 GPU snapshot at18:46UTC showed100% GPU
utilization and51769MiB allocated device memory; this is a point sample, not
whole-run utilization. Smoke peak torch memory38.308GiB each.

Expected comparison around September15 03:45–04:30UTC (September14 23:45–
September15 00:30EDT; September15 12:45–13:30KST), conditional on observed
full-run speed and no failures. Research may finish after the submission archive;
it is not a dependency of that delivery. No new standalone research heartbeat was
created: combined clear-ec-ssl is ACTIVE every30minutes with separate completion
conditions and unchanged submission download/verification duties.

## Complete — September15 03:51:33UTC CPU audit

Both queues finished successfully: fold2 at03:49:25UTC (32644.15seconds),
fold3 at03:50:01UTC (32678.79seconds). Summed queue runtime18.15GPUh,
including smoke/evaluation/controller overhead, not billed node-hours.
Both GPU pods areCompleted and releaseGPU; only the separate all-fiveSSL two
GPUs remain training. No new GPU or retry was started.

Frozen `audit` ran once on a GPU-free bounded CPU helper after bothcompletion
markers. Source/input hashes, all finalcheckpoint hashes, exact splits/IDs/targets,
5215updates,raw-target policy,strictreload records and same three-batch hashes
passed. The two intermediateSSL checkpoints still have6000-step schedule metadata
and validation_images_used=0. Metric tests zero/NaN/emptycolumn boundaries passed.
Aggregatecomparison and twoqueuecompletion files are copied to local
results/ssl_duration_20260914. Image-level data/OOF/checkpoints stay onPVC.

Same3572 primary images, valid CD3572/CV3572/HEX3570. Lower is better:

| Initialization | CD | CV | HEX | Mean | Fold2 mean | Fold3 mean |
|---|---:|---:|---:|---:|---:|---:|
| Raw/no new SSL |7.026769|10.671788|10.365762|9.354773|9.333034|9.376437|
| SSL2000 |6.985229|10.714271|10.315628|9.338376|9.341321|9.335497|
| SSL4000 |6.927192|10.697520|10.319176|9.314629|9.330873|9.298519|
| SSL6000 control |6.874325|10.692191|10.311389|9.292635|9.323447|9.262036|

Fold2-only choice between2000/4000 is4000. Research gateFAILS:4000 mean is
0.021994 worse than6000, not>=0.03better, and bothfolds are worse than6000.
2000 is0.045741 worse than6000. Therefore no early-checkpoint promotion,
newall-datafit,calibration or packaging. Existing sixth-onlyrelease and all-five
SSL fits continue using their alreadyfrozen6000-stepencoder unchanged.

This is a single-model comparison on reused developmentfolds, not an independent
test or six-memberensemble score. It does not validate the separate all-five
SSL replacement candidate. Durationbranch is nowterminal/successfullyaudited;
its negative researchgate is not a jobfailure. Distillation remainsqueued until
the newer all-fiveSSL twoqueues areterminal and its ownteacher-evidencegates pass.
Do not repeatdurationaudit or recreatefinishedGPUjobs.
