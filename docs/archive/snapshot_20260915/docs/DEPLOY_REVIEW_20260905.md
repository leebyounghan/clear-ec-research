> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../DEPLOY_REVIEW_20260905.md) for the release-facing summary.

# September 5 — corrected initialization and submission numerical audit

## Current decision

### Completion checked September 5, 23:19 UTC

Corrected all-data fit finished22:47:29 UTC:3,874 updates in11,604 seconds
(3h13m24s). Postflight finished22:51:00 with no failures. Separate staged
candidate4/5 artifacts,32 native image/candidate/precision checks and both
MHA-to-three-JSON entrypoints succeeded. `package_manifest.json` records exact
files and timing scope. This is NOT a built container, competition-device test,
upload or submission. The later region-search queue remains active; do not
delete the shared monitor just because this deployment queue completed.

GPU0 has been idle since postflight ended; region-search's frozen scheduler
is restricted to GPUs1–7. Do not overwrite its source or start its pending
fold manually into the same output path (the scheduler would reject it).

User authorized implementation after the code/validation review. Preserve all
prior OOF, raw data, all-data weights and staged artifacts. No challenge upload,
submission, pod creation/deletion or unrelated file cleanup is authorized here.

Hybrid completed **19:31:18 UTC September 5**, all five folds successful.
`results/hybrid_20260905_audit_v2.json` contains the corrected comparison:

| Fixed ensemble | Raw OOF | Exploratory cross-fit calibration |
|---|---:|---:|
| Existing three + metric (four members) | 9.045733 | 9.017020 |
| Existing three + hybrid (four members) | 9.036248 | 9.009273 |
| Existing three + metric + bag (five members) | 9.035524 | 9.004594 |

Keep the five-member candidate. The hybrid is not an accuracy upgrade over it;
it can be reconsidered as a smaller fallback only after measuring target runtime.
The revised triage compares BOTH current4/current5, requires raw and calibrated
improvement and worst-fold delta <= +0.02. It never automatically authorizes
deployment. This is a revised review criterion, not the original screen rule.
Old audit files and checkpoints remain unchanged; the new audit uses `_v2` names.

## Changes and tests

- Model construction accepts a model class and creates heads exactly once.
  Prediction-bag all-data training now matches the OOF initialization sequence.
  The old double-initialized fit remains preserved, not relabeled as corrected.
- Submission loading checks LM/LS and rejects feature-pooling bags. Candidate
  membership/order, view batch and positive finite three-element calibration
  are checked before loading any weights.
- Added explicit output root and trusted same-run resume. Every 1,000 updates,
  `last_training.pt` stores optimizer, scheduler, Python/NumPy/Torch/CUDA RNG and
  exact epoch/batch position. Resume is opt-in and checks config/source hashes.
  Older weights-only runs cannot be resumed with this mechanism. Recovery may
  roll back to the last 1,000-update checkpoint; no automatic restart is enabled.
- Controller status is refreshed every polling cycle. Old completed controllers
  are not restarted merely to update their historical status files.
- Eight focused CPU tests passed, including initialization/RNG identity, exact
  interrupted-update recovery, strict metadata and hybrid-reference rejection.
  Four existing candidate-compatibility tests and seven method-training tests
  also passed: **19 total**. A missing legacy interface test fixture in the
  isolated snapshot was copied from the preserved source before rerunning the
  four tests. Runtime source hashes were independently rechecked: zero mismatch.
  All8,954 OOF IDs resolve to slide groups for the bootstrap postflight.
- Corrected one-update GPU training smoke passed. Both new recipes passed
  BF16/FP16 submission-code smoke on two held-out images. The initial smoke
  incorrectly applied a whole-fold score threshold to two images; this was
  restricted to full audits. Prediction drift gates were not relaxed. Example
  smoke drift was <0.027% per prediction but aggregate two-image MAPE shifted
  -0.0188 points. Full-fold score drift tolerance remains absolute 0.01 points.

## Active bounded work

### Update: September 5 20:35 UTC

All five full-fold numerical audits completed successfully at20:14:30–36 UTC,
about40m30s–40m36s each. Across both new recipes and all8,954 held-out images,
BF16 predictions exactly reproduce saved OOF values (max difference zero).
FP16 maximum individual relative difference is0.757787%; the maximum absolute
member/fold raw-score delta is0.001389 points. Peak allocated memory per resident
sequential member is3.124922GiB BF16 and2.641036GiB FP16. These are not full
container cold-start timings or target-device certification. Ensemble/cluster
bootstrap aggregation remains in the automatic postflight after fitting.

The corrected fit remains healthy at1200/3874updates (3622.73s, finite loss).
GPU0 is training; GPUs1–7 are idle. Auditing GPUs1–5 have been idle for about
21minutes at this check. No extra work was launched to fill them. The existing
fit/postflight monitor remains active; no candidate is replaced or submitted.

Started **19:33:58–19:34 UTC September 5** on the existing one-node eight-H200 pod.
Root: `/workspace/clear-ec-research/experiments/deploy_review_20260905`.
Frozen source: `source/code`; do not overwrite it while workers are running.
Manifest records source hashes. Top-level original pod code is untouched.

| GPU | Work | Launch PID |
|---|---|---:|
| 0 | Corrected predbag2 all-data fit, 3,874 frozen updates | 62217 |
| 1–5 | Full numerical audits, folds 0–4 respectively | 62223/62229/62235/62241/62247 |
| 6–7 | Unassigned | — |

The fit uses the same 8,998 valid rows, targets, batch48 and update-index cosine
schedule. It does NOT tune against all-data training loss. Each audit covers both
new recipes, both precisions, all held-out rows: **8,954 × 2 × 2 = 35,816
image/member/precision evaluations**, each with 40 views. Old three-seed audits
are reused only after checking exact row/target/ID alignment.

Full numerical gates: BF16 max relative prediction error <= 1e-5; FP16 mean
relative error <= 0.5%, max <= 5%, absolute whole-fold raw score drift <= 0.01
points. The five folds run independently; failed audits prevent staging.

At 19:36 UTC, fit reached40/3874 in129s; each audit processed300 images in122s.
These early rates project roughly **3.5 hours** for the corrected fit and
**45–55 minutes** for each parallel fold audit. Prior same-size bag fit took
3h14m. Plan for fit plus postflight around **23:00–23:20 UTC September 5**
(September 6 08:00–08:20 KST), subject to measured throughput/checkpoint I/O.
GPU6/7 are idle; do not represent this as eight occupied GPUs.

## Automatic postflight and limits

`run_deploy_review.py` stages only after the fit and all five audits succeed:

1. Reconstruct four/five-member BF16/FP16 OOF ensembles, report score drift and
   conditional paired slide bootstrap. This does not remove historical selection
   bias and is not new independent validation.
2. Copy old three seeds + unchanged metric weights + corrected bag weights to
   this run's separate staging directory, using SHA256-verified copies.
3. Keep the existing frozen calibration scales. No all-data model score is used.
4. Test eight fixed raw MHA images with four/five members in both precisions,
   alternating precision order. Record per-image wall time and peak allocation.
5. Test actual MHA/three-JSON entrypoint in fresh subprocesses; emit
   `inference_summary.json`, `package_manifest.json`, and `complete.json`.

These are native H200 checks, NOT Docker/network-isolated/target-device checks.
Runtime Dockerfile pins the observed PyTorch2.11.0 CUDA13 base by registry digest
and the observed core package versions. It has NOT been built. CUDA13 driver
compatibility with the competition host is unverified; official challenge pages
returned403 during this check. Do not assume generic T4 documentation establishes
this phase's exact limits.

Local Docker is available but runs on ARM64; local disk had only26GiB free.
Model files alone are about7.33GB before runtime, unpacked layers and build cache.
User was asked for an external volume or existing Linux builder. No user files,
Docker images or caches were removed; no giant image build/download was started.

## Follow-up

The OpenAI Docs-guided 30-minute thread monitor `clear-ec` was recreated for this
specific queue. It stays quiet on unchanged healthy progress, reports actionable
completion/failure, and deletes itself after completion. Scheduling is local and
can arrive late; record the actual remote time separately. The remote controller
advances the finite queue and postflight independently of the desktop monitor.
Official guidance: https://learn.chatgpt.com/docs/automations?surface=app

After native completion: resolve builder storage and target runtime, build/test
the real container offline, and obtain explicit authorization before any upload
or submission. Crop/seed/nested-performance experiments remain separate future
decisions; this queue does not start them automatically.
