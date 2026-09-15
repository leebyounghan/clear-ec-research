> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../FINAL_CANDIDATE_20260904.md) for the release-facing summary.

# Final candidate — authorized Sep4 2026

User approved continuing with three-seed candidate inference/calibration validation
and all-data retraining. No submission or pod lifecycle action is authorized here.

## Execution

### COMPLETE: Sep4 15:04:35 UTC /Sep5 00:04:35 KST

All three25-epoch fits, five full-fold inference audits and automatic postflight
completed successfully. `complete.json` has8 done, no failures, no active/pending
jobs. `package_manifest.json` and `inference_summary.json` both exist. At the
15:24:29 UTC check all eight GPUs were idle, about20minutes after postflight.
No new work was added, no pod was stopped, and no challenge upload/submission was
performed. The completed-queue monitor was paused via the app, using OpenAI Docs.

Each fit completed4700updates. Seed42/77/2026 runtimes were6583.98/6607.52/6596.03s
(~1h50m each; summed fit-process runtime5.4965GPU-hours, not billed node-hours).
Primary selected checkpoints remain the precommitted3576/2831/2980updates,
not the final25-epoch backups. No score is assigned to the all-data fits.

Candidate files are staged on the pod at:
`/workspace/clear-ec-research/experiments/final_candidate_20260904/staged/`.
Four model files total4,889,690,037bytes (~4.89GB; excludes runtime/container).
Source hashes and all four staged model SHA256/size records were independently
rechecked successfully at completion. Original datasets and OOF artifacts remain.

| File | SHA256 |
|---|---|
| cpsam_v2 | 0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667 |
| seed42.pt | f9ba4e281d3044d4a9e17da84cb2846fcafff1891d37b301c4e9c5910a03df3c |
| seed77.pt | a89edd82c4ce59ca942b3214f151ac63c14641bfcc67ac9f09b1748d4e0caa76 |
| seed2026.pt | 174fdac5400442634aa571ad60ef86e8e693aee4f5b7ec3c7d862ee890d1bef5 |

The actual staged entrypoint processed an original MHA and wrote all three finite
scalar JSON files in both BF16 and FP16. Fresh-process wall time was12.687s BF16
and9.081s FP16 on H200, including startup/loading. These are single-image smoke
timings, with FP16 run second (filesystem caching can differ), not a controlled
precision-speed comparison or target-device certification.

**Remaining before submission:** build a compatible container; test with network
disabled and the real file interface; verify exact phase runtime/resources and
target-device memory/speed; stage/upload only when authorized, then explicitly
submit at the user's chosen time. The current pod has no container builder.
Do not call this artifact a built/uploaded submission container. The validated
OOF9.038264 result remains internal exploratory calibration, not an evaluation
of the new all-data candidate or a leaderboard forecast.

### Update: Sep4 14:38 UTC /23:38 KST

All five full-fold inference audits passed, completing14:13:05–14:14:04 UTC.
Across all8,954 held-out images and15 model/fold combinations, chunked BF16
member outputs exactly reproduce saved OOF predictions. FP16 maximum individual
relative prediction drift is0.863724%; the ensemble raw score changes only
9.073177 ->9.073457 and exploratory cross-fit score9.0382643 ->9.0383884.
The tiny raw BF16 difference from historical9.073195 is float32 ensemble mean
versus the historical audit's float64 averaging, not changed member predictions.

Mean resident three-member time on H200 is1.0863s/image BF16 or0.9255s/image FP16,
excluding model initialization/checkpoint loading. Peak allocated memory per
sequential member is3.125GiB BF16 /2.641GiB FP16. These are not cold-process or
T4 measurements; the final entrypoint smoke remains pending.

All three preselected all-data checkpoints now exist. Fits are at step3560–3580,
epoch19–20 of25, with finite losses and zero failures, continuing the authorized
fixed schedule. GPU0–2 are training; GPUs3–7 are idle after their completed audits
(about24minutes by this check). Do not claim all8 GPUs are still busy or start
extra experiments. Completion/postflight estimate remains around15:10 UTC,
subject to actual remaining epochs. No pod shutdown or submission occurred.

### Launch snapshot

Verified live at13:14 UTC /22:14 KST: all three training smoke tests passed;
the three full fits started13:13:41 UTC and all five audits started13:13:20 UTC.
Eight active jobs, zero failures, no pending jobs. Training has finite losses
through20 optimizer updates; each audit has processed100 held-out images.
Memory snapshots: ~26,009 MiB per trainer and ~3,963 MiB per auditor. GPU
utilization fluctuates with preprocessing/loading; do not describe every instant
as100% compute utilization.

- Pod: `clear-ec-h200-0`, container `researcher`, one node with eight H200s.
- Controller: `code/run_final_candidate.py`, PID54446 at launch around 13:11 UTC
  (22:11 KST) Sep4. It prepares data, starts eight bounded GPU jobs, and performs
  postflight staging and actual MHA/JSON entrypoint tests automatically.
- Root: `experiments/final_candidate_20260904`; logs: `logs/final_candidate_20260904`.
- GPU0/1/2: all-data training seeds42/77/2026, each after a one-step smoke test.
- GPU3–7: fold0–4 held-out inference audits, each covering three trained OOF
  members in both BF16 and FP16 on every held-out image. This is numeric parity,
  timing and memory validation, not five new trained models.
- Locks, source hashes, completed-output guards and bounded timeouts prevent
  duplicate workers and uncontrolled continuation. Smoke limit10min, job limit8h,
  postflight limit30min. No job is added merely to occupy a GPU.

## Data and training selection

Separate cache: `datasets/clear_ec_final9000_20260904` (9,000 central512 originals).
Existing 8,954 pixels and labels must match before the manifest is published.
All 46 recovered file/CSV aliases are included. Two rows have nonpositive HEX:
as in the validated OOF recipe, exclude those rows before slide averaging.
Thus there are **8,998 eligible training images**, not 9,000 usable metric triples.
The original raw archive, CSVs and OOF caches/folds remain untouched.

Use CPSAM mean pooling, per-metric log-space heads, fixed normalization, slide-
smoothed targets, random native256 within central512, BS48, BF16, AdamW encoder
1e-5/head1e-3, weight decay1e-4, 200-update warmup and gradient clipping1.0.
All trainable parameters, including LayerNorm, are in the optimizer.
The old `train_final9000.py` and old submission entrypoint are not reused or changed:
their checkpoint layouts/training details differ from the validated OOF model.

Each final fit continues for25epochs and preserves every fifth epoch. The primary
checkpoint is chosen **before training** by median best OOF optimizer updates:

| Seed | Five best OOF epochs | Fixed selected update | Approx final epoch |
|---|---|---:|---:|
| 42 | 24,25,18,19,24 | 3576 | 19.02 |
| 77 | 22,22,18,19,16 | 2831 | 15.06 |
| 2026 | 20,23,20,21,19 | 2980 | 15.85 |

OOF used149 updates/epoch; final uses188 updates/epoch (including the short last
batch so every eligible image participates). This preserves selected optimizer
update counts more closely than copying epochs despite the larger training set.
The partial last batch is an explicit difference from OOF drop-last. Epoch25 is
a separately preserved fallback, **not chosen using in-sample accuracy**. Full-data
retraining does not itself have an unbiased validation score.

## Calibration stability already checked

Source: `calibration_audit.json`, using the pre-existing aligned OOF predictions.
Equal model weights, no fitted member selection weights.

| Cross-fit correction strength | Internal overall |
|---|---:|
| None | 9.073195 |
| Half | 9.045847 |
| Full | 9.038264 |

Both half and full strength improve all five folds relative to no correction.
The slide-bootstrap conditional95% interval for full-minus-none is
[-0.045573,-0.023575]. It is conditional on fitted models and historical reuse,
not fully nested uncertainty and not a leaderboard guarantee.

The pre-frozen candidate scales fit on all OOF rows are CD0.9975358743,
CV0.9850454674, HEX0.9888038017. This full OOF fit is for deployment; its training
fit score is not independent validation. Calibrating new all-data models may
transfer imperfectly; keep uncalibrated/half-strength options documented.

## Inference contract and postflight

`final_candidate_core.py` and `submission_candidate_20260904/inference.py` are a
separate candidate path. CPU unit tests cover exact OOF model state/output
compatibility, exact crop/normalization/D8 parity, weighted-median calibration and
the MHA/JSON interface with a mocked predictor. The latter does not substitute
for GPU inference.
All new files compile. GPU smoke/parity completion is tracked in the run artifacts.

The entrypoint loads each model strictly, uses40 views per image (C5xD8) in batches
of8, averages encoded predictions within each model, decodes, then averages the
three models in physical target units and applies pre-frozen scales. The old
checkpoint names cannot silently load via `strict=False`. Images are uint8
grayscale MHA; unsupported shapes or duplicate input files fail explicitly.

Full-fold audits compare chunked BF16/FP16 against saved OOF predictions. They
require finite positive outputs, mean relative drift below0.5% and max below5%
per member as coarse failure gates; detailed score drift must still be reviewed.
Resident inference timing, checkpoint load time and peak allocated memory are
reported separately. These H200 measurements do **not** certify T4 speed or VRAM.

After all eight jobs succeed, postflight automatically:

1. Summarizes held-out BF16/FP16 metrics and inference timing.
2. Copies the three selected weights and CPSAM architecture weights to
   `staged/model`, records hashes and frozen settings.
3. Runs the real candidate entrypoint on an original MHA with BF16 and FP16 and
   checks three finite scalar JSON outputs, recording cold-process time.
4. Writes `package_manifest.json`, then controller `complete.json`.

This is native entrypoint testing and staging only. Docker build/network-isolated
container testing, exact challenge phase runtime verification, target-device
testing, upload and submission remain separate steps. Grand Challenge sets time
limits per phase; generic T4 documentation is not proof of CLEAR-EC's allocation.
[Runtime documentation](https://grand-challenge.org/documentation/runtime-environment/),
[submission workflow](https://grand-challenge.org/documentation/making-a-challenge-submission/).

Observed pod runtime: PyTorch2.11.0+cu130, Cellpose4.2.1.1, NumPy1.26.4,
OpenCV-headless4.11.0.86, SimpleITK2.5.5. No docker, podman, buildah or buildctl
binary is available inside the current pod. The old Dockerfile uses a different
PyTorch runtime and must not be assumed to reproduce this candidate without a
separate build/test. No dependency installation or environment replacement was done.

## Timing and monitoring

First measured progress (provisional, before the first full epoch):20 updates in
~28.69s =1.4345s/update. At that rate4700updates (25x188) take6742s or1h52m,
plus checkpoint I/O. The selected3576/2831/2980updates would be reached in
~85.5/67.7/71.2min respectively. From the13:13:41 UTC launch, provisionally
budget completion around15:10 UTC /00:10 KST Sep5, then postflight. Revise once
full epochs confirm throughput; this is not a promised deadline.

Audits initially took35.6–37.9s per100images per model/dtype. Extrapolating
1791images x3models x2dtypes gives~64–68min per fold, plus loads and warmups.
The five folds run concurrently. A resident three-model ensemble would imply
~1.1s/image on this H200 setup, but the actual cold-process integration timing
is still pending and FP16 throughput may differ. No T4 runtime is inferred.

The previous BS48 central OOF jobs took about2h for25 training+validation epochs.
Final fits have25% more training rows but no epoch validation; infer a provisional
2–3h budget, to be revised from actual epoch times. Full-fold FP16/BF16 audits are
larger than a smoke test; use the first100-image timing to estimate their completion.
Do not promise readiness before measured progress. Sep10/Sep14 remain user targets,
with earlier candidate freeze and container validation buffers.

The existing `clear-ec` 30-minute monitor was reactivated with this finite scope.
OpenAI Docs informed the monitor update; local checks need the computer/app
available. Remote training **and postflight** run independently of those checks.
The monitor must distinguish actual time from a delayed scheduled timestamp,
report real idle gaps and pause after all work is complete. Do not start extra
experiments, stop pods, contact others or submit on its own.
[Official scheduled-task guidance](https://learn.chatgpt.com/docs/automations?surface=app).
