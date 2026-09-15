# CLEAR-EC — raw import and submission preparation

> Sep4: User approved the next scope. See
> [FINAL_CANDIDATE_20260904.md](FINAL_CANDIDATE_20260904.md) for the three all-data
> fits, five inference audits and automatic postflight. The completed28-job queue
> below remains unchanged and must not be restarted.

## Completed audit — Sep4 12:54 UTC / 21:54 KST

All **28/28 jobs completed successfully**, including eight context GPU smoke tests.
Both source manifests match the code used to launch them. No pod lifecycle action
or challenge submission was performed. The completed-queue monitor was paused.

- Central sweep: completed Sep4 01:05:06 UTC / 10:05 KST, elapsed **6h31m45s**.
- Context queue: completed Sep4 05:28:05 UTC / 14:28 KST, elapsed **6h33m34s**
  from dependency release, including the staggered handoff from the prior queue.
- All eight GPUs were idle when actually checked at 12:54 UTC. The last queue
  completion was **7h26m earlier**. The heartbeat carried a Sep3 21:30 timestamp
  but this turn actually executed Sep4 12:54; the cause of that delay is unverified.
  Timely post-completion follow-up was missed. Do not claim continuous utilization.
- Epoch log totals: central sweep **40.049 GPU-hours**; context **28.192 GPU-hours**.
  These sum measured epoch train/validation durations, not billed node time, and
  exclude startup, smoke tests, checkpoint I/O and final D8. Each job logged 25 epochs.
- Central BS48 jobs: ~2.00h epoch runtime; BS24 native control: ~2.08h;
  BS24 two-view context jobs: ~4.00–4.01h, confirming roughly double encoder work.

### Final five-fold central result

Source: `experiments/submission_20260904/ensemble_audit_08_groups.json`.

| Method | Raw OOF | Exploratory cross-fit scale |
|---|---:|---:|
| Fixed baseline seed42 | 9.209081 | 9.177907 |
| Seed77 | 9.158686 | 9.144607 |
| Seed2026 | 9.181989 | 9.134550 |
| Attention pooling | 9.251321 | 9.218225 |
| Raw targets | 9.203243 | 9.163093 |
| Seed42 + seed77, equal weights | 9.095811 | 9.069605 |
| **Seeds42/77/2026, equal weights** | **9.073195** | **9.038264** |

The three-seed mean improves raw OOF by 0.135886 and exploratory calibrated OOF
by 0.139643 versus seed42. It improves the previous two-seed calibrated result
by 0.031341. Calibrated CD/CV/HEX are 6.556761 / 10.364384 / 10.193648.
This is repeated internal OOF, not fully nested model/calibration selection or a
leaderboard forecast. No new ranking claim follows from it.

### Full-frame hypothesis: current design did not help

Source: `experiments/context_20260904/screen_summary.json` (folds0/1 only).

| Method | Raw two-fold score | Delta vs native | Fold0 / fold1 |
|---|---:|---:|---|
| Central native control | 9.152901 | — | 9.321953 / 8.983849 |
| Central overview + native | 9.326950 | +0.174048 | 9.371590 / 9.282310 |
| Full overview + uniform native | 9.416639 | +0.263737 | 9.470011 / 9.363266 |
| Full overview + focus native | 9.400113 | +0.247211 | 9.508464 / 9.291762 |

Every context variant loses on both folds. Full-frame focus is only 0.016526
better than uniform and remains worse than native. Do not automatically expand
these variants to five folds. This tests a resized overview plus a native256 view
with bag attention; it does **not** rule out larger native crops, better ROI
localization, or every possible full-frame architecture. Central-context also
loses, so periphery alone is not an established explanation. Two-fold screening
and the aggregation change limit causal conclusions.

### Recommended next scope — not launched by this monitor

Prioritize the three-seed baseline as the current candidate, check inference
latency/container limits and calibration sensitivity, and prepare a separately
versioned all-9000 retraining set using the reconciled labels. A larger native
crop requires its own matched experiment; these context failures are not a reason
to repeat them automatically. No additional GPU jobs were started beyond the
authorized finite 28-job queue. Obtain direction for the next bounded run.

OpenAI Docs was used to manage the completed monitor. Official guidance says
local scheduled tasks require the computer on and app running; that requirement
does not establish the cause of this particular delay.
[Official scheduled-task guidance](https://learn.chatgpt.com/docs/automations?surface=app).

## Historical launch snapshot

Snapshot: 2026-09-03 20:49 UTC / 2026-09-04 05:49 KST. This supersedes the
historical action list in NEXT_STEPS.md. No pod was stopped and no submission was made.

## User-provided original images

- Local: `/Users/byounghanlee/Desktop/clear-ec-train9000-raw.tar.gz` (unchanged).
- Pod: `/workspace/artifacts/clear-ec-train9000-raw.tar.gz`.
- Bytes: 6,633,443,603.
- SHA256: `799a833e53f33196882d5e5b014f6309ec9c7d5250110093dda88a7dfb2fe263`.
- Archive contains 9,000 original MHA images and `final_train_ids.csv`.
- Separate import: `datasets/clear_ec_fullframe_20260904/`.
- Existing 8,954 cache IDs all matched. Their labels match the archive CSV and
  their central 512x512 pixels exactly match the raw originals.
- Full-frame cache: uint8 `[8954,972,1296]`, 11,279,461,248 bytes.
  Existing central cache, training rows and folds were not changed.
- Full-frame global256 inputs and unlabeled 4x6 grid focus scores are prepared.
  See `fullframe_manifest.json`, `context_views_manifest.json`, and `import_status.json`.
- Initial import safely rejected the extra root CSV; importer was updated to
  validate/preserve that CSV, then successfully resumed extraction. The retry1 logs
  are authoritative, not the initial failure log.

The remaining 46 raw IDs differ from CSV IDs only by whitespace/hyphens. The
separate `audit_raw_id_aliases.py` requires a unique bijection over all 9,000 IDs
and preserves source IDs. Its audit and `source_metadata/final_train_ids_9000_reconciled.csv`
are for future all-data retraining, **not a silent expansion of current OOF folds**.
An earlier exact-ID-only observation that these images lacked labels was incomplete.

## Running and queued experiments

Pod `clear-ec-h200-0`, container `researcher`, one node `h200-04-w-11f1`, eight H200s.
At 20:45 UTC all eight GPUs were 99% utilized (~28.5 GiB allocated each).
Memory capacity alone is not evidence that more simultaneous trainers improve throughput.

### Existing central512 sweep: 20 jobs

`code/run_submission_sweep.py` -> `experiments/submission_20260904`.
Four groups x five fixed slide-group folds: baseline seed77, seed2026, attention
spatial pooling, and raw targets instead of slide-smoothed targets. 25 full epochs,
BS48, CPSAM, fixed optimizer, identical split assignment; best checkpoint final D8.
At 20:45 UTC: 8 complete, 8 running, 4 pending, no failures.

First eight jobs completed in about 2h11m after the 18:33 UTC start. A simple
three-wave estimate is about 6h33m total (roughly 01:06 UTC / 10:06 KST Sep4),
subject to attention pooling, validation, I/O and later runtime differences.
This is an estimate, not a promised completion time.

### Raw context screening: 8 jobs

`code/run_context_screen.py` -> `experiments/context_20260904`.

| Group | Inputs per sample | Selection |
|---|---|---|
| central_native | One native256 from central512 | Uniform |
| central_context | Central512 resized256 + native256 | Uniform |
| full_context_uniform | Original whole frame resized256 + native256 | Uniform across original |
| full_context_focus | Original whole frame resized256 + native256 | Contrast/sharpness heuristic |

Each uses folds0/1, seed42, BS24, 25 full epochs. Two-view models use bag attention;
the single-view control uses mean pooling. Native views retain cell scale; overview
views supply global context. This is not full-resolution whole-frame encoding.
Focus selection uses image pixels, not ROI annotations or metric labels.
Train/evaluation bag sizes match, and D8 is averaged across independent bags.

Controller PID at launch: 51795. Data dependencies are complete. It starts when
the prior sweep's pending list is empty, using only GPUs absent from that sweep's
active list and from actual CUDA processes. This can fill the otherwise idle four
GPUs during the last wave, without competing with pending prior jobs.
A fresh prior status is required. Old waiting-only controller and its manifest were
replaced; `manifest_wait_all_v1.json` preserves that revision. No training worker
or pod was stopped. Every new job must pass a one-step GPU smoke test before full
training. Full-frame GPU training has **not yet started at this snapshot**.

Budget: a context job has twice the encoder views of the matched native control;
with BS24, neither the BS48 baseline time nor a precise 4h estimate is established
yet. Provisionally allow ~4–5h for two-view jobs and revise after measured epochs.
Two-fold results are screening only; advance a winner to all five folds before
container selection, subject to observed benefit and remaining time.

## Latest internal results

`ensemble_audit_05_groups.json` uses aligned 8,954 OOF predictions. Lower is better.

| Method | Raw OOF | Exploratory cross-fit scalar calibration |
|---|---:|---:|
| Fixed baseline | 9.209081 | 9.177907 |
| New baseline seed77 | 9.158686 | 9.144607 |
| Fixed baseline + fixed JEPA, equal weights | 9.137417 | 9.094570 |
| Fixed baseline + seed77, equal weights | 9.095811 | 9.069605 |

Calibration is meta cross-fitting, not fully nested base-model validation. The
same historical validation data has been reused. These are **not leaderboard
forecasts**, nor proof of rank. Two invalid HEX targets are masked per metric.
No ensemble member weights were fitted in these simple pairs.

## Checks and operational limits

- Nine tests passed: context shape/augmentation, unchanged model/fold code,
  bounded jobs and safe queue handoff; safe import and synthetic pixel/label
  matching; submission manifest and held-out calibration isolation.
- Actual archive hash, raw pixel and label comparison, full-frame preprocessing
  completed successfully. GPU context smoke tests remain queued.
- Finite 20+8 jobs; controller locks, immutable manifests, completed-output
  protection, bounded smoke/train timeouts. Do not restart duplicate workers.
- Existing 30-minute `clear-ec` monitor follows both queues and reports meaningful
  changes only. Scheduled local checks depend on the desktop environment being
  available; remote queue execution does not depend on a monitoring turn.
- No submission, external sharing/contact, new pod, pod deletion or shutdown.

## Submission buffers

Target first submission Sep10, final Sep14. Aim to freeze candidates Sep8/Sep12
and upload/validate containers Sep9/Sep13 or earlier. Treat Sep14 as the public
Phase-I deadline; do not assume Sep15. Exact cutoff timezone must still be checked.
Do not depend on the first submission result arriving before final preparation.

Previously checked official sources:
[UPenn challenge](https://opticlab.med.upenn.edu/clear-ec-challenge),
[challenge site](https://clear-ec.grand-challenge.org/),
[submission workflow](https://grand-challenge.org/documentation/making-a-challenge-submission/).
