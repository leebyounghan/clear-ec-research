> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../REGION_SEEDS_20260906.md) for the release-facing summary.

# Native crop seed replication — updated September 8, 00:50 UTC

## GPU pod removed at user request; persistent workspace retained

On September 8 the user explicitly requested shutting down the GPU pod if idle.
Live pre-deletion checks found all eight H200 GPUs at 0% utilization and 0 MiB,
no compute applications, and no active training/inference/controller processes
(only sleep infinity, historical zombies, and the diagnostic itself).
The workspace mount was verified as the Lustre-backed clear-ec-workspace PVC;
both regional selected.pt files, protected staged5, experimental stage6, and
datasets were present under that mount. Latest integration complete.json still
reported failed={} and deployment_approved=false.

Only pod clear-ec-h200-0 in namespace p-ajou-mlxp-project was deleted, with normal
graceful termination. kubectl wait --for=delete confirmed deletion by00:50UTC.
The PVC remained Bound to pvc-exa-abfcdbc5-269d-4f5c-a069-6716e1606c6a afterward.
No PVC, dataset, model, or experiment file was deleted. The pod's ephemeral
container state is gone; persistent workspace files can be reattached later.
No replacement pod was created. The completed experiment/Docker polling
automation clear-ec was deleted to stop obsolete recurring checks.

Docker was being used for submission-image packaging, not training, and the
user's Mac Docker is not inherently required: another compatible builder could
be used after its environment/authority are established. Do not treat local
Docker repair as a prerequisite for all further work. Container build, target
runtime/driver compatibility, final candidate selection, and actual submission
remain outstanding; no new builder or compute allocation is authorized here.

The September 7 status below is historical, not a claim that the pod still exists.

---

## Six-model native integration COMPLETE; Docker engine unavailable

`region_candidate_20260907/complete.json` completed07:11:50UTC without failures.
Both explicit candidate6 configs passed8native images x2precisions, malformed
configuration rejection, original-five-member equivalence, correct decoded
member averaging/calibration, and fresh-process MHA->exactly3finite-positiveJSON
outputs for each precision. Fresh outputs matched direct-call results.
All8GPUs were idle after completion. No training/audit/integration job is active.

Six-member OOF from the actual precision audit (not new independent validation):

| Branch(seed77) | BF16 raw | FP16 raw | Precision delta | Exploratory crossfit |
|---|---:|---:|---:|---:|
| anchor_quality | 9.017185416 | 9.017327933 | +0.000142516 | 8.987525987 |
| full_quality | 9.016114292 | 9.016263682 | +0.000149391 | 8.987421876 |

Warm Python process, includes per-image six-model construction/loading and
preprocessing,8fixedimages,concurrent two-GPU jobs (NOT directly controlled
timing comparison against old5 and NOT competition device):

- anchor meanBF16 8.388s / FP16 7.837s,peak3.125/2.641GiB.
- full meanBF16 8.370s / FP16 7.860s,peak3.125/2.641GiB.
- Fresh Python process wall time10.827–12.029s for the single entrypoint image,
  not cold container startup. Container and target device remain untested.

Read-only local builder check07:11UTC: Docker CLI29.6.1 darwin/arm64 exists,
contextdesktop-linux; endpointunix:///Users/byounghanlee/.docker/run/docker.sock.
`docker version` did not return a server after~40seconds; only that diagnostic
client PID17747 was terminated, NOT Docker backend or user processes.
Bounded socket GET /_ping returned exact text **Docker Desktop is unable to start**.
The pod has no docker/podman/nerdctl/buildctl on PATH. Macfree105GiB at07:10UTC.
No Docker restart/install/cache deletion/large download was attempted.

User action requested: open Docker Desktop and check the engine/startup error.
Next safe work: prepare/review an appropriate experimental Dockerfile/COPY list
and challenge runtime compatibility, then build only when a working builder is
available. Existing CUDA13/PyTorch2.11 recipe still needs target-driver checks.
No default regional branch is selected/promoted; no container built/uploaded or
submission made. Do not repeat finished experiments solely to occupy GPUs.

---

## Historical experimental integration launch (now complete)

`region_inference_20260907` completed06:53:52UTC, all six jobs passed, no failures.
Actual07:03:53UTC check: all8GPUs idle, about10minutes after last completion.
All8954 OOF images x2policies x2precisions x40views were checked. Recomputed native
views exactly match cached OOF views, and BF16 predictions match saved OOF exactly.
FP16 maximum relative difference is0.006764335(anchor,0.6764%) and
0.011564847(full,1.1565%); maximum absolute single-member fold MAPE drift0.001203532.
All predeclared numerical gates passed. Five full jobs took2684–2740s each,
about3.78GPUh combined (plus short smokes/all-data check).

Single-member forward timing excluding native preprocessing and weight loading:
BF16~0.345s/image,FP16~0.298s/image,H200,40views. Peak allocated memory3.125GiB
is the peak across precisions, NOT an independently reset FP16-only measurement.
Eight fixed all-data images also passed. Selected checkpoint SHA256:

- anchor77:279f7cdc6bb01062d1418bafded66b3b17fee9f7f9e169f76025128d35bd76c6
- full77:619a0263340da2d18ef837fc29306d00dff8162241e36d85edf9cf4fdd453f03

NEW `experiments/region_candidate_20260907`, preparation controllerPID89843.
This creates a separate `experimental_stage` with both explicitly named six-model
configs, not a replacement/default for protected staged5. It verifies all12
protected stage file hashes and regional audited hashes before copying. It combines
aligned actual BF16/FP16 OOF predictions, checks six-model precision score drift,
and fits three scalar calibrations to historical BF16 OOF with fixed equal member
weights. This calibration is not independent/nested validation or a model-weight search.

Integration jobs will use GPU0(anchor77) andGPU1(full77) on eight fixed raw images,
both precisions, including actual fresh-process MHA->threeJSON entrypoints.
Checks include fail-closed malformed configs, exact averaging/calibration formula,
and agreement of the five old members with protected candidate5 after removing its
own calibration. Single-member encoded predictions are decoded before member mean;
the old calibrated ensemble is NEVER mixed directly with uncalibrated regional outputs.

Source: local `code/region_candidate_entrypoint.py` and
`code/prepare_region_candidate.py`, copied to isolated experiment/source. Monitor
controller.log/PID during preparation (failure can precede status.json), then
stage_manifest.json,status.json,integration/{policy}/progress.json/result.json,
and complete.json. No automatic rerun. Each integration worker has1800s timeout,
each fresh entrypoint300s. GPU2–7 are unused by these checks.
No Dockerfile is copied because the old COPY list and default config would be
wrong for this stage. Actual container build, runtime selection, competition GPU
compatibility, final branch selection and upload/submission remain outstanding.

---

## Historical native inference audit launch (now complete)

All-data fits are COMPLETE, not active: anchor77 completed05:17:20UTC at4470
updates (13,040.63s), full77 completed05:40:18UTC at4917updates (14,418.46s).
Combined training time7.628GPUh. Controller completion05:40:27UTC, no failures.
Actual check06:03:55UTC found all8GPUs idle, about23.5min after last completion.
Do not restart either all-data fit. Both selected.pt remain in
`experiments/deploy_region_20260907/runs/{policy}`; protected staged5 unchanged.

NEW `experiments/region_inference_20260907`, controller PID88317, started06:07UTC.
Six finite GPU jobs: GPU0–4 folds0–4, GPU5 all-data native smoke. At06:08:26UTC
all six initial two-image smokes had passed and full jobs were running with
no failures. Full PIDs fold0=89413,fold1=89419,fold2=89074,fold3=89080,
fold4=89086,all_data=89425. Read status.json for live state. GPU6–7 are unused;
GPU5 will become idle after the short eight-image all-data check completes.

New `region_inference_core.py` recomputes quality scores/origins directly from a
native image (no training-cache lookup). Each OOF image's40 normalized nativeD8
views must exactly match frozen NativeRegionViews before prediction. Both seed77
policies are loaded strictly and predicted through single-crop CandidateNet,
averaged in encoded space then decoded exactly as OOF. Two precisions BF16/FP16.
Five full audits cover all8954 OOF rows, after each two-image smoke passes.
Gate unchanged from prior deployment audit: BF16 maxrelative<=1e-5; FP16
maxrelative<=.05, meanrelative<=.005 and absolute full-fold MAPE drift<=.01.
These are numerical parity checks, not independent model-performance tests.

Sixth job verifies both new all-data checkpoints on fixed8native images (first4
shared IDs + first4 extra IDs), BF16/FP16 finite-positive output and precision
parity. Its two-image smoke uses one shared/one extra image. No training-set
accuracy is calculated. All-data samples are for interface/numerics only.

Each smoke timeout1200s, full audit10800s, upper allocation18GPUh combined.
Controller never retries/overwrites, checks source hashes and free GPUs before
launch, and only terminates its own timed-out process groups. Source under
this new experiment/source, manifest.json, status.json, logs/, smoke/, audits/
and eventual complete.json record provenance and progress. Existing frozen
training code, weights and staged candidate remain unchanged.

Completion still does NOT certify a submission: actual combined six-model
entrypoint/configuration, calibration, container build and competition-device
driver/runtime tests remain before any candidate promotion or submission.

---

## Historical all-data launch snapshot (01:40 UTC; now complete)

`experiments/deploy_region_20260907`, controller PID87549, launched about01:38UTC.
Only two GPU jobs, GPU0 anchor_quality and GPU1 full_quality, both seed77 (the
first replication seed, deliberately not the numerically best seed2026).
Both smoke PIDs87615/87621 completed successfully by01:39:51UTC. Each passed
exact head-initialization equality against frozen OOF ScalarNet, all9000 native
ID/label/central pixel alignment, shared8954 quality origins and fixed D8 view
equality, then one finite optimizer update and selected-checkpoint save.
The controller launched full-fit PID87866 (anchor GPU0) and87872(full GPU1).
At01:39:51UTC full processes were still in startup/data checks; no full-update
progress had yet been measured. Read progress.json for actual step throughput.

The new adapter reuses the frozen corrected `train_deploy_methods.py` through
three exact, guarded in-memory substitutions (seed, native-view data loader,
configuration provenance); training/optimizer/checkpoint logic stays unchanged.
The original frozen source is untouched. New source was edited locally with
apply_patch and transferred to an exclusive new remote source directory.

All9000 rows are mapped to the original8954 raw rows by exact ID plus46 already
extracted extra MHA files. Every central512 image is checked against the final9000
cache; shared labels and all shared quality origins must match, and a fixed
shared image's40 normalized D8 views must be exactly equal before training.
Only8998 positive-target rows are fitted, using existing all-data slide smoothing.
No extra11GB cache or external weights are downloaded/created for this mapping.

- anchor77 OOF best epochs[35,31,28,27,30] -> median30*149 =4470updates.
- full77 OOF best epochs[33,34,32,30,34] -> median33*149 =4917updates.
- Same5215-update cosine schedule, mixedMAPE, native256K2, BS48, BF16,
  originalCPSAM initialization, single head construction and fixed LM/LS.
- Each smoke gets1200s; each full fit21600s. Planning budget11GPUh combined.
  Approximate full-fit time3.5–4.5h based on previous all-data training, to be
  replaced with measured step timing. Six GPUs remain idle, not hidden.
- Controller checks source hashes before each launch, refuses occupied GPUs,
  stops only its own timed-out process groups, never retries failed runs, and
  stops after these two fits. `manifest.json` records source hashes, preceding
  audit hash, fixed epochs/update selection and budget; `status.json`, logs,
  run progress/result files and eventual `complete.json` are monitoring sources.
- Completion requires a separate native inference/precision/packaging audit.
  No all-data validation score, staged replacement, upload or submission.

The older all-data-preparation comments below are historical; inspect this new
queue rather than relaunching it or repeating the completed seed audit.

## Completed five-fold seed confirmation (latest)

All 12 confirmation fits completed September 7 at 00:00:53 UTC (09:00:53 KST),
without failures, after 37,955 seconds (10h32m35s). Actual remote inspection was
00:59:30 UTC although the delayed heartbeat carried a September 6 21:07 trigger.
At inspection all eight GPUs were idle: approximately 59 minutes since the last
fit completed. No new GPU workload has been launched by this audit.

Manual audit completed at 01:01 UTC; report and exact executed audit source:
`experiments/region_seedconfirm_20260906/manual_fivefold_audit_20260907.json`.
Local reusable source: `code/audit_region_seedconfirm.py`. Inputs were read-only;
the report was exclusively created, and input result/OOF/split SHA256 hashes
recorded. All 8,954 validation indices are unique, each fold is disjoint from its
training indices, IDs/targets/train splits agree with same-policy seed42 and
current5 members, and saved predictions agree with final D8 result metrics.
These checks do not eliminate historical validation reuse or checkpoint-selection bias.

| Added regional model | Seed | Five-fold raw add6 | Delta vs current5 | Confirmation folds2–4 delta | Worst fold delta | Gate |
|---|---:|---:|---:|---:|---:|---|
| anchor_quality | 77 | 9.017185427 | -0.018338795 | -0.019205661 | -0.010447109 | pass |
| anchor_quality | 2026 | 9.016603418 | -0.018920804 | -0.019244199 | -0.006591541 | pass |
| full_quality | 77 | 9.016114304 | -0.019409918 | -0.021537900 | -0.002607473 | pass |
| full_quality | 2026 | 9.014771307 | -0.020752915 | -0.022109847 | -0.006701835 | pass |

Current5 raw is 9.035524222. All four new variants improve every fold and all
three aggregate metrics, satisfying the predeclared gate. Delta against the same
policy seed42 is anchor77 -0.002363923, anchor2026 -0.002945931,
full77 -0.000190739, full2026 -0.001533736. Choosing the best seed after these
results adds selection bias; a numerically best seed is not independently proven superior.

Add6 CD/CV/HEX:

- current5: 6.538585685 / 10.382685967 / 10.185301015.
- anchor77: 6.514295193 / 10.363743228 / 10.173517859.
- anchor2026: 6.520633463 / 10.354709305 / 10.174467486.
- full77: 6.512107765 / 10.356560354 / 10.179674793.
- full2026: 6.509439877 / 10.355208120 / 10.179665925.

Fixed three-seed regional averages at total weight1/6 (eight actual models)
score anchor9.017097712 and full9.015150688; neither beats its best single-seed
add6. No weight optimization was performed. Extra inference cost is not yet
justified. Exploratory calibrated add6 scores respectively are anchor77
8.987525976, anchor2026 8.988884833, full77 8.987421875, full2026 8.987107212;
these are not nested validation or leaderboard forecasts.

Next work is bounded all-data-fit/inference preparation for an eligible regional
branch, with training budget and checkpoint step selection based on its fold
metadata, then native inference/precision and packaging verification. No new
all-data fit, staged replacement, container build, upload or submission has
been done in this audit. Keep the protected five-model staged candidate intact.

---

## Historical snapshot: September 6, 13:29 UTC

All results below use historically reused slide-group OOF, with validation
checkpoint selection. They are not independent tests or leaderboard forecasts.
No candidate was deployed, uploaded or submitted. Existing five-model staged
artifacts, original data, models and frozen training source remain preserved.

## Completed original search

`experiments/region_search_20260905` completed at 07:31:48 UTC: 14 supervised
fits plus one external SSL fit, no failures. Current5 raw five-fold MAPE is
9.035524222; exploratory cross-fit calibrated score is 9.004594466.

| Added native crop model, seed42 | Five-fold raw add6 | Raw delta | Exploratory calibrated add6 |
|---|---:|---:|---:|
| anchor_quality | 9.019549349 | -0.015974873 | 8.990568197 |
| full_quality | 9.016305043 | -0.019219179 | 8.987590419 |

Anchor improves all five folds. Full-quality worsens fold3 by 0.001062746.
Both four-model replacement alternatives are worse than current5.
Full-uniform failed its two-fold gate. Native external SSL improved add6 by
only 0.003614080, below the predeclared 0.01 threshold; no further SSL folds.
This does not rule out all domain pretraining methods.

## Completed seed screen

Remote root: `/workspace/clear-ec-research/experiments/region_seedcheck_20260906`.
Eight fits (two policies × seeds77/2026 × folds0/1) completed at 12:58:40 UTC,
after 19,093.77 seconds (5h18m14s), no failures. All eight GPUs were idle at
13:26:18 UTC when completion was observed. New full fits began at 13:28:38 UTC,
so the last-finishing GPU had about 30 minutes between training queues.

Manual audit at 13:27 UTC verified identical indices, IDs, targets and training
splits against seed42 and all current5 members, disjoint train/validation,
and saved D8 prediction/metric agreement. The full audit command and report
are preserved remotely in `seed_audit.json` (exclusive new output).

Comparison: add6 = (5 × current5 + new prediction) / 6, no optimized weights.
Current5 on these same 3,582 rows is 8.992169590, NOT its five-fold score.

| Policy | Seed | Add6 raw | Delta vs current5 | Delta vs policy seed42 | Fold0 / fold1 delta |
|---|---:|---:|---:|---:|---|
| anchor_quality | 42 | 8.977828645 | -0.014340945 | 0 | -0.007948 / -0.020734 |
| anchor_quality | 77 | 8.975129901 | -0.017039689 | -0.002698744 | -0.010447 / -0.023632 |
| anchor_quality | 2026 | 8.973732697 | -0.018436893 | -0.004095948 | -0.010784 / -0.026090 |
| full_quality | 42 | 8.974909196 | -0.017260395 | 0 | -0.010364 / -0.024157 |
| full_quality | 77 | 8.975950250 | -0.016219341 | +0.001041054 | -0.007077 / -0.025362 |
| full_quality | 2026 | 8.973451309 | -0.018718281 | -0.001457887 | -0.011559 / -0.025877 |

Add6 CD/CV/HEX respectively:

- Current5: 6.372011842 / 10.376060944 / 10.228435985.
- Anchor77: 6.352365010 / 10.348719012 / 10.224305682.
- Anchor2026: 6.351267584 / 10.343540431 / 10.226390077.
- Full77: 6.351252610 / 10.347338742 / 10.229259397 (HEX slightly worse).
- Full2026: 6.349305863 / 10.345144162 / 10.225903903.

Both policies pass the predeclared replication gate: each new seed improves
both folds and aggregate raw delta <= -0.01. Equal seed42/77/2026 regional
averages at fixed total weight1/6 score 8.974997879 (anchor) and 8.974305167
(full). These require EIGHT actual models, not six; their tiny incremental
improvements over seed42 do not yet justify the inference cost.

## Active bounded confirmation

Remote root: `/workspace/clear-ec-research/experiments/region_seedconfirm_20260906`.
Controller PID84387, launched 13:28 UTC. Frozen 120-source SHA256 checks passed;
no training source was edited. Manifest stores complete controller/launch
commands, inherited hyperparameters, source hashes and preceding audit hash.

Twelve fits: two policies × seeds77/2026 × folds2/3/4. At 13:29:07 UTC eight
smokes passed, eight full fits were running with GPU utilization100% at that
snapshot, about51,769MiB each; four fold4 jobs await free GPUs.

| GPU | Initial full job | PID |
|---|---|---:|
| 0 | anchor77 fold3 | 85332 |
| 1 | anchor2026 fold3 | 85333 |
| 2 | full77 fold3 | 85334 |
| 3 | full2026 fold3 | 85335 |
| 4 | anchor77 fold2 | 85336 |
| 5 | anchor2026 fold2 | 85337 |
| 6 | full77 fold2 | 85338 |
| 7 | full2026 fold2 | 85339 |

Unchanged native256, two crops, batch48,35epochs, mixed-MAPE/cosine and original
CPSAM initialization. Each job first passes one-update/two-image smoke;
12h full-job and15min smoke limits; no restart or automatic additional fits.
Fold4 has1,790 validation rows; other folds have1,791. Final split alignment
must match same-policy seed42. Queue stops after all12 jobs.

Manual confirmation criterion recorded BEFORE these fits: each seed77/2026
must have confirmation delta <= -0.01, five-fold delta <= -0.01 and all its
five fold deltas negative. Report CD/CV/HEX and worst-fold results even when
the aggregate improves. Reconstruct five-fold predictions from original seed42,
seedcheck folds0/1, and this queue's folds2/3/4; do not refit blending weights.
No two-fold result or queue completion automatically replaces staged current5.

Measured first wave: approximately5.3GPUh/fit including final inference.
12 × 5.3 =63.6GPUh; budget66GPUh with overhead. Two sequential waves project
10.6–11h wall time, finishing September7 around00:05–00:30 UTC (09:05–09:30 KST),
with09:00–10:00 KST a practical range. After the first8 finish, only four jobs
remain; four GPUs will be idle unless a separately justified bounded task is
prepared. Do not hide idle capacity or launch duplicate work solely to fill it.

Long exploration cutoff remains September8 23:59 UTC; reserve September9 for
any selected all-data fits and container/target-device validation before the
September10 candidate. September14 is the second requested submission target.

## Packaging and local storage

Protected `deploy_review_20260905` completed September5 22:51 UTC, including
corrected all-data prediction-bag fit and native precision/entrypoint checks.
Twelve staged file hashes were verified earlier. Actual container build,
competition hardware/driver test, upload and submission are still NOT done.

Mac free space was109MiB at07:37 UTC and local writes had failed earlier.
At13:28 UTC read-only `df` now reports26GiB free (95% used); no files were
deleted by this agent and the cause of recovered space is unknown. Small local
documentation updates are now possible. This is not proof that full container
build capacity or an appropriate builder/target driver is available. Do not
download large weights or install/delete build tools implicitly.

The existing30-minute monitor is being updated under OpenAI Docs guidance to
follow this new finite confirmation queue, preserving quiet healthy checks.
