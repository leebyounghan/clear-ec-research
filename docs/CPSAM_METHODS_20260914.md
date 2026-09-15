# Same-backbone DINO / I-JEPA comparison — September 14, 2026

## User intent and corrected scope

Continue work here from `01a0559f-96bc-7932-b65e-267cc8d747d5`. User clarified that
the current CPSAM backbone should be updated with the DINO and I-JEPA **methods**;
do not substitute the papers' official pretrained backbones. The earlier official
DINO ViT-B/8 experiment is a different experiment, not a result of this comparison.
No official I-JEPA model weights were downloaded or trained in this task.

All branches start from the same `cpsam_models/cpsam_v2`, matching the current
raw/standard-SSL research initialization. This is the existing 304,542,720-parameter
CPSAM encoder: 24 blocks, width 1024, native patch8, 32x32 learned positions, global
attention with learned relative positions, original 256-channel inference neck.
No branch starts from an all-data supervised regression checkpoint, which would
contaminate the fold comparison with validation labels. Downstream uses the
unchanged CandidateNet CD/CV/HEX head, one encoder, and existing 40-view TTA.

The user also welcomed distillation and asked about a cell-aware expert teacher.
The immediate controlled comparison remains DINO versus I-JEPA. A cell-expert
ablation is a research proposal, not an automatically launched extra GPU wave.

## Methods

- **DINO on CPSAM:** image-level self-distillation using mean pre-neck patch features,
  the upstream DINO projection-head implementation (random initialization), 8192
  prototypes, bottleneck256, hidden2048; centered/sharpened EMA teacher, temperatures
  student0.1/teacher0.04, center momentum0.9. Two native256 globals and two native128
  local views from the same field; no pixel resizing. Local views interpolate
  positional embeddings, not pixels. Prototype gradients are frozen for 200 steps.
- **I-JEPA on CPSAM:** target teacher sees the full native256 field. Student patch
  tokens are gathered BEFORE all attention; target positions never enter its encoder.
  Learned SAM absolute AND relative positions retain their original coordinates.
  Upstream I-JEPA multi-block masks: context scale .85–1, four targets each .15–.2,
  aspect .75–1.5, disjoint context/targets, minimum10. Upstream transformer predictor,
  width384/depth6/heads12, smooth-L1 against feature-normalized teacher targets.
  Predictor is discarded after training. No added variance/covariance loss is used.
- Both retain and freeze the inference neck during SSL, then train it normally in
  downstream regression. Neither changes the encoder state-dict architecture.
- Both: all 9000 images without label filtering; 6000 updates x batch32, identical
  epoch-permutation image-index stream; warmup200, cosine LR floor.05; AdamW wd.04;
  EMA .996→1, gradient clip1, BF16, block activation checkpointing; mild D8, gain±5%,
  offset±.02. No strong scale, blur, or solarization augmentation.
- Encoder learning rates 5e-6 and 1e-5 for each method. DINO head multiplier100;
  I-JEPA predictor multiplier10. This is a limited learning-rate search, not a claim
  of global hyperparameter optimality. Equal updates/exposures do not equal FLOPs.
- Native preprocessing remains the existing CLAHE/percentile pipeline. Masked-pixel
  independence tests apply to preprocessed encoder inputs, not raw-image preprocessing.

Sources: [DINO](https://github.com/facebookresearch/dino),
[I-JEPA](https://github.com/facebookresearch/ijepa),
[I-JEPA paper](https://arxiv.org/abs/2301.08243).
I-JEPA code commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`; DINO code comes from
the previously verified commit `7c446df5b9f45747937fb0d72314eb9f7b66930a`.
These are adaptations of the objectives to CPSAM, not reproductions of ImageNet runs.

## Comparison and candidate gate

1. Four independent SSL trials: `dino_low`, `dino_high`, `ijepa_low`, `ijepa_high`.
   Each passes a real-GPU four-update smoke and strict checkpoint reload before its
   fresh full run. Save terminal teacher and feature diagnostics; extract 5-view
   mean pre-neck 1024-dimensional features for 9000 images to PVC only.
2. Fixed ridge alpha10, feature standardization fitted on fold2 training only, raw
   log targets; evaluate fold2 primary rows. Handle nonpositive targets separately
   per metric, matching the existing metric policy. Freeze one LR per method using
   fold2 mean MAPE only. Never consult fold3 for this selection.
3. Four independent full downstream fits: each chosen method x folds2/3. Existing
   raw labels, head initialization/seed77, K2/batch48, encoder1e-5/head1e-3, fixed5215
   updates, 40-view evaluation. Match first-three-batch hashes against raw controls.
   Strictly reload final model, verify IDs/targets/splits/checkpoint hashes.
4. Report standalone CD/CV/HEX and their mean on the same 3572 primary images,
   alongside the existing raw and standard-SSL controls. Choose the method using
   full downstream fold2 mean only. Require pooled mean at least0.03 below standard
   SSL AND improvement over both controls on each fold before candidate training.
5. If the gate passes, one all-data research fit from the chosen SSL teacher, with
   8998 positive-label rows, unchanged full_quality head recipe and5215 updates.
   Save the single-model checkpoint in this new experiment's candidate directory.
   Verify load/inference compatibility. Existing submission artifacts are untouched;
   no publication, submission, archive replacement, or Git push is part of this run.
   If the gate fails, report it and stop rather than inventing an improvement.

All9000 SSL includes all internal validation images WITHOUT their labels. Thus the
new results are **transductive**, not independent held-out validation. Historical
raw/standard controls used fold-clean training; comparisons with them include this
extra unlabeled image exposure. Fold2 is reused for tuning, fold3 is a historically
reused confirmation fold. No official score/leaderboard prediction is justified.

## Cell-aware teacher proposals

DINO uses an EMA of its student, not an independently trained cell expert. Our
CPSAM initialization already provides a cellular-segmentation prior. A next teacher
experiment should add demonstrably useful information, rather than just copying it.

- Patch-level distillation (iBOT-like) combines local masked targets with global DINO
  consistency; potentially useful for retaining cell boundaries and arrangement.
- Frozen cell-segmentation teacher can provide boundary/center maps and reliable
  instance regions. First audit corneal-image segmentation quality. Exclude uncertain
  regions; compare feature-only versus boundary-aware distillation. Segmentation
  predictions are pseudo-targets, not ground truth; segmentation-derived areas or
  neighbors are not automatically equivalent to clinical CD/CV/HEX labels.
- For a combined objective, ablate each term before combining it. Use the successful
  base method, add a small validated morphology term, then check for CD/CV/HEX tradeoffs.
  Teacher and auxiliary heads remain training-only: one student at inference.
- The earlier simple frozen spatial-relation anchor did NOT improve over standard
  SSL (9.335322 versus9.292635 on the same3572 rows). Do not repeat that result as a
  promising success or assume a frozen teacher guarantees an improvement.

References: [iBOT](https://arxiv.org/abs/2111.07832),
[Cellpose-SAM](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1).

## Execution and continuation

Namespace `p-ajou-mlxp-project`, PVC `clear-ec-workspace`.
Remote root `/workspace/clear-ec-research/experiments/cpsam_methods_20260914_v2`.
Local repo `/Users/byounghanlee/Documents/kubectl/clear-ec challenge/clear-ec-research`.
All model/data computation and tests run on K8s. Local work is code/orchestration
and aggregate JSON only; no images, labels, features, OOF predictions or weights.

Each Job owns ONE H200, 6CPU, memory request24Gi/limit64Gi. First stage has at most
four concurrent GPUs: DINO timeout8h each, I-JEPA timeout5h each. Downstream has four one-GPU Jobs, each timeout7h.
Candidate stage has one GPU, timeout7h. Runtime caps sum to61 GPUh if all stages run;
Pod deadlines/termination add overhead. Jobs have no retries and a10-minute TTL.
GPUs are released independently. No monetary cost claim. Do not relaunch deleted
Jobs merely because TTL removed them: inspect PVC completion/failure files first.

Commands below are intentionally staged. NEVER launch a later stage before its
requirements pass. Frozen runtime sources must not be edited after GPU launch.

### Stage1 launch

Local: `python3 code/launch_cpsam_methods_20260914.py ssl > k8s/cpsam-methods-ssl-0914-v2.json`
then `kubectl create -f k8s/cpsam-methods-ssl-0914-v2.json` once.
Jobs: `ce-cpsam-{dino,ijepa}-{low,high}-0914-v2`.
Results: `ssl/{tag}/smoke_complete.json`, `pretrain/{progress,diagnostics,result}.json`,
`feature_progress.json`, `probe.json`, `complete.json`.

### Stage2 after ALL FOUR Stage1 completions

Use an existing task-owned CPU helper or create the bounded no-GPU helper from
`k8s/clear-ec-methods-prep-0914.yaml` (only remove/recreate this helper if it expired).
Inside it, working directory `/workspace/clear-ec-research`:

`/workspace/venv/bin/python experiments/cpsam_methods_20260914_v2/source/run_cpsam_methods_20260914.py select`

Run select only if `selection.json` does not already exist. Verify source/checkpoint
hashes and identical sample-order hashes. Selection and scores are aggregate JSON.
Then locally generate/inspect `k8s/cpsam-methods-downstream-0914-v2.json` with the launch
script's `downstream` argument, and `kubectl create -f` it ONCE, provided no matching
Job, downstream run directory or failure already exists. Jobs are
`ce-cpsam-{dino,ijepa}-f{2,3}-0914-v2`. Each automatically performs its own downstream
smoke then full run. Results: `downstream/{method}_fold{fold}/{result,complete}.json`.
Remove only the task's temporary CPU helper when finished.

### Stage3 after ALL FOUR downstream completions

On a task-owned no-GPU CPU helper, run the frozen script's `audit` command. Verify
`comparison.json` and the reports, not just an exit code. Copy aggregate JSON locally
under `results/cpsam_methods_20260914`, leaving all individual arrays on PVC.
If `update_gate_passed=false`, report the comparison and pause the heartbeat.
If true, generate a candidate manifest using the launch script's `candidate --method`
with exactly `chosen_by_fold2`, and create it once after confirming its Job and
candidate directory have never existed. One Job `ce-cpsam-{method}-candidate-0914-v2`.
Do not switch winners using fold3. On completion verify the checkpoint SHA256,
fresh-model strict loading, finite prediction on a few fixed native views in K8s,
one encoder and compatible old inference-head keys. Save an aggregate verification
JSON, report results and the PVC checkpoint path, update this document, then pause.

### Monitoring boundaries

Follow up in the CURRENT task, not the old broken task. Read the actual UTC clock.
Stay quiet on unchanged/non-actionable status. Notify on meaningful stage change,
failure, required action, or final comparison/candidate completion. On failures,
read `failures/*.json` and progress/Job status; do not automatically retry, alter
frozen sources, or launch a substitute sweep. Existing independent Jobs can finish
within their own caps; report their state. Always inspect PVC when a Job is absent.
No extra GPU teacher/combined-objective experiment is authorized by this staged plan.
Absolute continuation deadline:2026-09-15 04:00UTC; after it, report state and pause.

## Verification and launch status

CPU tests: sparse full-token attention equals original SAM with NONZERO relative
position tables; masked-input perturbation invariance and zero hidden-input gradient;
EMA update/teacher detach; upstream DINO head/loss gradients; upstream I-JEPA predictor
mask alignment and encoder gradients. Five tests passed. Metric zero-target test
passed separately. Existing input provenance and9000-view alignment passed preflight.
Final source freeze and real-GPU smoke/throughput are recorded below when available.


## Preflight correction history

Initial four Jobs `ce-cpsam-{dino,ijepa}-{low,high}-0914` stopped BEFORE any SSL
updates at the full-encoder numerical equivalence test. TF32 was enabled in the
existing runtime; equivalent attention contractions had a maximum absolute
GPU difference0.00028156. A real304M-parameter CPU comparison returned exactly0.0.
The revised check disables TF32 for equivalence testing only, preserving the
original tolerance; normal BF16 training settings are restored afterward.
Original sources/manifest/failure records are preserved under
`experiments/cpsam_methods_20260914`. The active corrected wave is the `_v2` root
above; do not mistake initial failures for v2 failures. No original run is resumed.
A single one-GPU,30-minute preflight Job validates both methods before launching
v2's full four-trial wave. Account for this and the failed initial preflights in
allocated GPU time separately. The heartbeat is paused during this diagnosis.


## Corrected GPU preflight passed; v2 wave launched

02:34:45UTC: both methods completed four real-GPU updates, finite gradients/loss,
strict teacher reload and healthy diagnostics. Both real-encoder equivalence checks
returned maximum absolute difference0.0 in FP32, with the ORIGINAL tolerance.
DINO smoke measured3.583seconds/update; I-JEPA0.720seconds/update. These are short
smoke timings, not final full-run throughput. DINO therefore receives an8-hour
stage1 timeout (about6h SSL at this initial speed plus extraction/overhead);
I-JEPA retains5h. Model/loss/data settings are unchanged. Only the orchestration
limit and provenance manifest were finalized after this preflight, before any
v2 full SSL run. The earlier failed wave remains immutable.

Final v2 manifest SHA256:
`9de09024e64b89fb9e374ff05e181e9c57d2d88a87ec50171886bb65984ef222`.
The four v2 Stage1 Jobs have now been created. Pending actual full-run progress.
The continuation heartbeat is `cpsam-dino-i-jepa`, attached to the CURRENT task
`01a09da7-cf2f-7ca2-8550-06ee227582d6`; resume it on the v2 root and the04:00UTC
September15 continuation deadline after checking successful launch. The original
thread's paused `clear-ec-ssl` automation should stay untouched.


02:39:31UTC launch snapshot: all four v2 smokes passed; no v2 failure records.
DINO low/high25/6000,3.553/3.551s per update; I-JEPA low/high150/6000,
0.745/0.747s per update. Both methods use the same304M CPSAM initialization and
same image exposure schedule, but computational costs differ substantially.
Initial extrapolation: I-JEPA SSL about75minutes; DINO about5h55minutes, plus
feature extraction and I/O. Complete downstream comparison provisionally11–13h
from02:37UTC launch; gate-passing candidate fit adds roughly4–5h. These estimates
must be updated from actual throughput, not treated as completion promises.
Heartbeat `cpsam-dino-i-jepa` is ACTIVE on v2 with the September15 04:00UTC deadline.
Aggregate launch/provenance snapshots are in local `results/cpsam_methods_20260914`.
The task-owned CPU helper is cleaned up after launch. GPU Jobs run independently.


## 03:11UTC routine check

Actual readback03:11:44UTC: all four v2 Jobs Running, no failures, immutable
manifest/source hashes verified. DINO low/high550/6000 (~3.54s/update); I-JEPA
low/high2775/6000 (~0.736s/update). Each has encountered all9000 image IDs.
Latest diagnostics (DINO500; I-JEPA2500) remain within the fixed stopping limits;
these are collapse checks, not evidence of downstream performance improvement.
No stage transition, new GPU allocation or selection was performed. Existing
11–13h comparison estimate is unchanged. Aggregate snapshot:
`results/cpsam_methods_20260914/status_20260914_0311.json`.


## 03:43UTC routine check

Actual readback03:43:06UTC: all four v2 Jobs Running, no failures, immutable
manifest/source hashes verified. DINO both1100/6000 (~3.54s/update), I-JEPA
both5350/6000 (~0.732s/update). All have seen9000 images; latest diagnostics
(DINO1000/I-JEPA5000) remain within the fixed stopping thresholds. No completed
Stage1 reports yet, so no selection or later-stage allocation. No user notification
beyond the routine check, because the experimental stage is unchanged. Snapshot:
`results/cpsam_methods_20260914/status_20260914_0343.json`.


## 04:15UTC — both I-JEPA Stage1 trials complete

Verified at04:15:20UTC: both I-JEPA trials completed6000 updates, all9000 images,
strict reload, feature extraction and fold2 ridge evaluation. Their Jobs have
expired after completion and no longer allocate GPUs. PVC checkpoint/feature
hashes and ALL manifest input/source hashes were reverified. Same sample-order
hash across both trials; no v2 failures. DINO low/high are still training,
1625/6000 at04:13UTC; no Stage2 Jobs or selection file were created.

| I-JEPA encoder LR | CD | CV | HEX | Mean |
|---|---:|---:|---:|---:|
| 5e-6 | 7.447485 | 11.490374 | 10.837338 | 9.925066 |
| 1e-5 | 7.453071 | 11.663931 | 10.981016 | 10.032673 |

These are FROZEN-feature ridge HPO scores on fold2 primary1781 images
(valid target counts1781/1781/1779), not final full-regression model scores.
All validation images participated in unlabeled SSL; no comparison with the
historical fully finetuned9.292635 or official score is warranted. Lower LR
leads this limited I-JEPA screen; formal per-method selection waits for all
four Stage1 completions as specified.

Each I-JEPA SSL took4390.2–4390.6seconds (~73m10s). Job start to aggregate
completion was ~82m46–47s per one-GPU Job, approximately2.759 GPUh total for
both; this is an allocation estimate excluding unobserved termination overhead,
not the SSL-only total of~2.439 GPUh. Inputs and checkpoint details are in
`results/cpsam_methods_20260914/ijepa_verified_20260914_0415.json`; timing is in
`ijepa_timing_20260914_0415.json`. Continue waiting for DINO, with no new work.

## 04:49UTC routine check

Actual readback04:49:22UTC: both DINO Jobs Running with zero restarts,
2225/6000 updates (~37.1%), ~3.544s/update, all9000 images encountered.
Latest diagnostics at2000 remain within the fixed stopping limits. Immutable
manifest/source hashes verified; no v2 failures. Both I-JEPA completions remain
available on PVC. Selection is absent as expected; no later-stage Jobs launched.
No meaningful stage change or user action is required. Aggregate snapshot:
`results/cpsam_methods_20260914/status_20260914_0449.json`.

## 05:20UTC routine check

Actual readback05:20:11UTC: both DINO Jobs Running with zero restarts,
2725/6000 updates (~45.4%), ~3.562s/update, all9000 images encountered.
Latest diagnostics at2500 remain within the fixed stopping limits. Manifest
and frozen source hashes verified; no v2 failures. Both I-JEPA completion
records remain available. Selection is absent; no new GPU Jobs were started.
No actionable stage change. Aggregate snapshot:
`results/cpsam_methods_20260914/status_20260914_0520.json`.

## 05:51UTC routine check

Actual readback05:51:12UTC: both DINO Jobs Running with zero restarts,
3250/6000 updates (~54.2%), ~3.569s/update, all9000 images encountered.
Latest diagnostics at3000 remain within the fixed stopping limits. Manifest
and frozen source hashes verified; no v2 failures. Both I-JEPA completion
records remain available. Selection is absent; no later-stage Jobs launched.
No actionable stage change. Aggregate snapshot:
`results/cpsam_methods_20260914/status_20260914_0551.json`.

## 06:22UTC routine check

Actual readback06:22:14UTC: both DINO Jobs Running with zero restarts,
3750/6000 updates (62.5%), ~3.579s/update, all9000 images encountered.
Latest diagnostics at3500 remain within the fixed stopping limits. Manifest
and frozen source hashes verified; no v2 failures. Both I-JEPA completion
records remain available. Selection is absent; no later-stage Jobs launched.
No actionable stage change. Aggregate snapshot:
`results/cpsam_methods_20260914/status_20260914_0622.json`.

## 06:53UTC routine check

Actual readback06:53:14UTC: both DINO Jobs Running with zero restarts,
4275/6000 updates (71.25%), ~3.580s/update, all9000 images encountered.
Latest diagnostics at4000 remain within the fixed stopping limits. Manifest
and frozen source hashes verified; no v2 failures. Both I-JEPA completion
records remain available. Selection is absent; no later-stage Jobs launched.
No actionable stage change. Aggregate snapshot:
`results/cpsam_methods_20260914/status_20260914_0653.json`.

## 07:24UTC routine check

Actual readback07:24:13UTC: both DINO Jobs Running with zero restarts,
4800/6000 updates (80%), ~3.578s/update, all9000 images encountered.
Latest diagnostics at4500 remain within the fixed stopping limits. Manifest
and frozen source hashes verified; no v2 failures. Both I-JEPA completion
records remain available. Selection is absent; no later-stage Jobs launched.
No actionable stage change. Aggregate snapshot:
`results/cpsam_methods_20260914/status_20260914_0724.json`.

## 07:55UTC routine check

Actual readback07:55:16UTC: both DINO Jobs Running with zero restarts,
5325/6000 updates (88.75%), ~3.575s/update, all9000 images encountered.
Latest diagnostics at5000 remain within the fixed stopping limits. Manifest
and frozen source hashes verified; no v2 failures. Both I-JEPA completion
records remain available. Selection is absent; no later-stage Jobs launched.
No actionable stage change. Aggregate snapshot:
`results/cpsam_methods_20260914/status_20260914_0755.json`.

## 08:26UTC routine check

Actual readback08:26:13UTC: both DINO Jobs Running with zero restarts,
5825/6000 updates (~97.1%), ~3.582s/update, all9000 images encountered.
Latest diagnostics at5500 remain within the fixed stopping limits. Manifest
and frozen source hashes verified; no v2 failures. Both I-JEPA completion
records remain available. DINO still needs its final updates, checkpoint
verification, feature extraction and probe; selection is absent as expected.
No later-stage Jobs launched or actionable stage change. Aggregate snapshot:
`results/cpsam_methods_20260914/status_20260914_0826.json`.

## 08:58–09:00UTC — Stage1 verified; Stage2 launched once

Both DINO trials finished6000 SSL updates and feature/probe evaluation at08:45:19UTC.
All four Stage1 Jobs have now expired and released their GPUs. At08:58:11UTC,
the frozen `select` command completed successfully: ALL manifest input/source
hashes, all four terminal checkpoint/feature hashes and the common image-order
hash were verified. Every terminal encoder passed strict reload. No v2 failures.

| Method | Encoder LR | Fold2 frozen ridge mean | Selected |
|---|---:|---:|---|
| DINO | 5e-6 | 10.262162 | No |
| DINO | 1e-5 | 10.185160 | Yes |
| I-JEPA | 5e-6 | 9.925066 | Yes |
| I-JEPA | 1e-5 | 10.032673 | No |

These are learning-rate screening results, not full-regression performance.
I-JEPA leads this frozen-feature comparison; the downstream comparison remains
pending. Selection is frozen in PVC `selection.json`, copied as aggregate JSON
locally. All four saw the same image-index stream, including validation images
without labels; the transductive caveat remains.

DINO SSL took21501.7seconds per trial (~5h58m22s); Job start to aggregate completion
was~6h08m22–23s per trial. Across all four Stage1 trials, SSL-only time totals
14.3845 GPUh, while Job-start-to-aggregate allocation estimate totals15.0383 GPUh.
The latter excludes unobserved termination overhead and initial/preflight Jobs.
Timing aggregates are in `results/cpsam_methods_20260914/stage1_timing.json`.

After confirming no downstream Job, run directory, smoke directory or failure
existed, exactly FOUR Stage2 Jobs were created at08:58:43–44UTC:

- `ce-cpsam-dino-f2-0914-v2`, pod `ce-cpsam-dino-f2-0914-v2-nvzvl`
- `ce-cpsam-dino-f3-0914-v2`, pod `ce-cpsam-dino-f3-0914-v2-wzx99`
- `ce-cpsam-ijepa-f2-0914-v2`, pod `ce-cpsam-ijepa-f2-0914-v2-nrp4t`
- `ce-cpsam-ijepa-f3-0914-v2`, pod `ce-cpsam-ijepa-f3-0914-v2-nmw4z`

Each owns1 H200, timeout7h, no retries; allocation/start records are in
`downstream_launch_v2.json`. At08:59:51UTC, ALL FOUR two-update/four-image smokes
completed with strict reload and full5215-update run configurations were written.
No failures. DINO uses `dino_high`, I-JEPA uses `ijepa_low`, both folds; all other
downstream settings remain matched. Records: `stage1_complete_20260914_0858.json`,
`selection.json`, `downstream_start_20260914_0900.json` in the local aggregate folder.

Continue monitoring THESE Stage2 Jobs/PVC records. Do not repeat select or create
Stage2 again. Wait for all four downstream completions, then follow Stage3 audit
and gate instructions above. The temporary no-GPU helper is cleaned up after launch.

## 09:31UTC routine Stage2 check

Actual readback09:31:16UTC: all four downstream Jobs Running with zero restarts,
625/5215 updates (~12.0%), 2.949–2.951s/update and peak GPU memory38.31GiB.
Manifest/frozen source hashes verified; no v2 failures or downstream completions.
No audit, candidate or additional GPU work launched. Current throughput implies
roughly4h16m for full training per Job, plus final40-view evaluation and I/O;
this is an estimate, not a completed runtime or performance result.
No actionable stage change. Aggregate snapshot:
`results/cpsam_methods_20260914/downstream_status_20260914_0931.json`.

## 10:02UTC routine Stage2 check

Actual readback10:02:13UTC: all four downstream Jobs Running with zero restarts,
1275/5215 updates (~24.4%), 2.942–2.947s/update and peak GPU memory38.31GiB.
Manifest/frozen source hashes verified; no v2 failures or downstream completions.
No actionable stage change, audit, candidate or additional GPU work launched.
Aggregate snapshot:
`results/cpsam_methods_20260914/downstream_status_20260914_1002.json`.

## 10:33UTC routine Stage2 check

Actual readback10:33:13UTC: all four downstream Jobs Running with zero restarts,
1875/5215 updates (~36.0%), 2.967–2.969s/update and peak GPU memory38.31GiB.
Manifest/frozen source hashes verified; no v2 failures or downstream completions.
No actionable stage change, audit, candidate or additional GPU work launched.
Aggregate snapshot:
`results/cpsam_methods_20260914/downstream_status_20260914_1033.json`.

## 11:04UTC routine Stage2 check

Actual readback11:04:16UTC: all four downstream Jobs Running with zero restarts,
2525/5215 updates (~48.4%), 2.960–2.962s/update and peak GPU memory38.31GiB.
Manifest/frozen source hashes verified; no v2 failures or downstream completions.
No actionable stage change, audit, candidate or additional GPU work launched.
Aggregate snapshot:
`results/cpsam_methods_20260914/downstream_status_20260914_1104.json`.

## 11:35UTC routine Stage2 check

Actual readback11:35:15UTC: all four downstream Jobs Running with zero restarts,
3125/5215 updates (~59.9%), 2.970–2.972s/update and peak GPU memory38.31GiB.
Manifest/frozen source hashes verified; no v2 failures or downstream completions.
No actionable stage change, audit, candidate or additional GPU work launched.
Aggregate snapshot:
`results/cpsam_methods_20260914/downstream_status_20260914_1135.json`.

## 12:06UTC routine Stage2 check

Actual readback12:06:15UTC: all four downstream Jobs Running with zero restarts,
3775/5215 updates (~72.4%), 2.965–2.967s/update and peak GPU memory38.31GiB.
Manifest/frozen source hashes verified; no v2 failures or downstream completions.
No actionable stage change, audit, candidate or additional GPU work launched.
Aggregate snapshot:
`results/cpsam_methods_20260914/downstream_status_20260914_1206.json`.

## 12:37UTC routine Stage2 check

Actual readback12:37:16UTC: all four downstream Jobs Running with zero restarts,
4375/5215 updates (~83.9%), 2.971–2.973s/update and peak GPU memory38.31GiB.
Manifest/frozen source hashes verified; no v2 failures or downstream completions.
No actionable stage change, audit, candidate or additional GPU work launched.
Aggregate snapshot:
`results/cpsam_methods_20260914/downstream_status_20260914_1237.json`.

## Final comparison — 13:41UTC; update gate FAILED

All four Stage2 fits completed5215 updates, strict checkpoint reload and40-view
evaluation by13:29:06UTC. All matched their historical raw-control first-three-batch
hashes. Kubernetes Job completion times13:28:58–13:29:12UTC were captured before TTL
removal. No v2 failures. A transient API timeout affected only waiting for the CPU
helper; the following PVC read succeeded. No training retry or source change occurred.

At13:41:03UTC the frozen `audit` completed successfully. It verified manifest
input/source hashes, all eight new/control model checkpoint hashes and strict reload
records, exact OOF IDs/indices/targets against reference splits, and32 artifact hashes.
The following results use the SAME3572 primary rows (valid CD/CV3572, HEX3570).
They supersede unfiltered1791-row fold2 metrics in individual run logs.

| Single-model initialization | CD MAPE | CV MAPE | HEX MAPE | Mean |
|---|---:|---:|---:|---:|
| Raw CPSAM | 7.026769 | 10.671788 | 10.365762 | 9.354773 |
| Historical standard SSL | 6.874325 | 10.692191 | 10.311389 | **9.292635** |
| CPSAM + DINO | 6.998434 | 10.687293 | 10.322082 | 9.335936 |
| CPSAM + I-JEPA | 6.989277 | 10.656864 | 10.311434 | 9.319192 |

| Initialization | Fold2 mean (1781 rows) | Fold3 mean (1791 rows) |
|---|---:|---:|
| Raw CPSAM | 9.333034 | 9.376437 |
| Standard SSL | 9.323447 | **9.262036** |
| DINO | 9.297394 | 9.374306 |
| I-JEPA | **9.296308** | 9.341990 |

Fold2-only method selection chose I-JEPA, with a small0.001086 advantage over DINO
on that tuning fold. Overall I-JEPA is0.016745 better than DINO and0.035581 better
than raw, but0.026556 WORSE than standard SSL. DINO is0.043301 worse than standard.
Both improve fold2 versus standard but regress on fold3. I-JEPA CV improves while
CD worsens versus standard; HEX is essentially unchanged. Do not infer a robust
new best backbone from this limited two-LR/single-seed experiment.

`chosen_by_fold2=ijepa`, `update_gate_passed=false`. The required overall mean
threshold was9.262635 or lower plus improvement over both controls on each fold.
Neither condition is met. NO Stage3 all-data candidate was launched, and the
existing model/submission remains unchanged. All9000 SSL was transductive and
included validation images WITHOUT labels; historical controls were fold-clean.
These results do not establish independent validation or an official-score gain.

Stage2 training took15468.98–15482.21seconds per Job (~4h18m), evaluation688–689s
(~11m29s). Total training17.1944 GPUh; evaluation0.7651 GPUh; Job start→completion
allocation estimate18.0233 GPUh, including smoke/verification/I/O. Combined v2
Stage1+Stage2 allocation estimate33.0617 GPUh, excluding original failed preflights,
the dedicated GPU preflight and unobserved Stage1 termination overhead.

Final aggregates: `results/cpsam_methods_20260914/comparison.json`,
`downstream_status_20260914_1331.json` (actual PVC check13:40:19UTC),
`downstream_job_completion.json`, `downstream_timing.json`. Checkpoints/OOF remain
only under the new PVC experiment root. The no-GPU helper is cleaned up and the
current-task heartbeat is paused after reporting this gate-failed comparison.
Any additional hybrid or cell-teacher experiment requires a new scope; none ran.
