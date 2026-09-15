# All five remaining members initialized from standard SSL — September15

## Completed — September 15

Both queues succeeded: queue0 at07:46:52UTC (18306.871287 seconds), queue1
at08:06:16UTC (19471.042152 seconds). Total10.49 runtime GPU-hours includes
smoke/checks/controller overhead, not billed node-hours. At the post-deadline
check both training pods were absent, consistent with their TTL cleanup.

Frozen CPU audit completed at2026-09-15T13:48:12+0000: source/input and five selected
checkpoint hashes verified; selected steps and strict-reload records passed.
Aggregate [completion.json](../results/all_members_ssl_20260915/completion.json)
contains all five hashes. Checkpoints remain at
`/workspace/clear-ec-research/experiments/all_members_ssl_20260915/runs/<member>/selected.pt`.
No independent accuracy score, ensemble packaging, promotion or new GPU was added.
The operational entries below are historical; do not recreate these queues.

## Explicit user decision

User requested all other models to be trained with SSL and explicitly chose
"기존 CPSAM + SSL 가중치로 5개 재학습". This keeps CPSAM architecture and uses
the completed all9000 standard custom masked EMA encoder. It is NOT DINO/I-JEPA
architecture replacement, five new independent SSL pretrains, or five copies of
the already trained sixth regressor. Fresh heads are constructed separately.

Existing official files and the delivered sixth-only SSL archive remain immutable.
No official submission, Git push, calibration fit, packaging or promotion is
performed by this training queue. Full-data fits have no independent validation
score. All-five replacement has NOT been validated by the earlier sixth-only
9.019599 comparison. Shared initialization can reduce ensemble diversity.

## Preserved recipes

Only encoder initialization changes. Original central512/native256 crop,
slide-smoothed Y_train labels, seeds, losses, learning rates, optimizer, schedules
and precommitted checkpoint steps are retained. First-five raw-label conversion
is NOT included. Existing sixth is SSL+raw/full_quality and remains unchanged.

| Member | Selected update | Training loop | Queue |
|---|---:|---|---:|
| seed42 |3576|original25epochs/4700updates, logL1, K1|1|
| seed77 |2831|original25epochs/4700updates, logL1, K1|0|
| seed2026 |2980|original25epochs/4700updates, logL1, K1|1|
| metric35 |4470|mixedMAPE, K1, original5215-step cosine trajectory|1|
| predbag2_35 |3874|prediction-bag logL1, K2, original5215-step cosine trajectory|0|

The inherited selected steps were chosen for original non-SSL recipes, not
optimized for SSL. Seed25epoch backups are NOT selected by training loss.
Eligible supervised rows8998, SSL actual eligible/seen9000, SSLupdates6000.
SSL SHA256:971bcaba9f66cbf9466b6e4bfd3e5e26f593ef56384dee1a2368133647bded32.

## Implementation and checks

Local new code: code/all_members_ssl_20260915.py; prior trainers untouched.
Remote root:/workspace/clear-ec-research/experiments/all_members_ssl_20260915.
Frozen source copies include wrapper, train_final_candidate,train_deploy_methods,
final_candidate_core,method_training_core. Local/remote SHA copies match.
Manifest SHA256:6f2adfaa489709c67afba25ea05eec8ac8eac01c841a9d1509507b004e548e13.
Wrapper SHA256:61450ecc9be916225e42e69492878a9ece8b20ea7dab88e6f6c04b041e8c2c03.

CPU preflight passed on K8s:9000/8998 data and finite/positive eligible targets,
allsource/input hashes, SSL strict encoder restore and exact tensor equality,
fresh head hash unchanged by encoder injection, torch CPU RNG unchanged,
finite parameters. Source is frozen before launch; no live-source mutation.

Each member has an independent1update GPU smoke, strict checkpoint reload and
four-image40view finite-positive inference. Then a fresh process starts full fit,
followed by the same reload/inference guard. These are numerical/interface checks,
not efficacy measurements or actual archived Docker/T4 validation.

## Jobs and bounds

Created September15 02:41UTC, namespace p-ajou-mlxp-project, PVCclear-ec-workspace.
Two new1H200 Jobs,6CPU/request24Gi/limit64Gi each:

- clear-ec-allssl-q0-0915: predbag2_35 then seed77.
- clear-ec-allssl-q1-0915: metric35 then seed42 then seed2026.

Per-queue runtime23400seconds(6.5h), outer23700, Job25200 including scheduling,
backoff0, TTL600. Maximum new training runtime13GPUh. No automatic retries,
duplicate jobs, new members or further hyperparameter sweeps. Each queue exits
after its own last verified fit; it does not holdGPU for another queue or audit.
Existing duration GPUs continue untouched; CLEAR-EC total at most4 for this wave.
At creation pods were initializing, not yet claimed training.

Historical ~1h50 eachseed, ~1h47metric, ~3–3.5hpredbag imply roughly5–6h wall
from actual start on twoGPUs, not a guarantee. User-supplied official deadline
September15 06EDT=10UTC. There are about7h18 at launch; packaging, download,
platform import and performance validation are not included in this fit estimate.
Do not promise official submission readiness. Delivered sixth-only archive remains
available independently. All model/data computation stays on K8s.

## Follow-up

Read queue{0,1}/{status,complete,failure}.json, queue logs, and
{smoke,runs}/{member}/{config,progress,result,verified}.json. Missing jobs may
have expired viaTTL: inspectPVC first, never recreate a trainingJob automatically.
Bothcomplete/no failure -> run frozen wrapper `audit` ONCE on a bounded CPU helper
if completion.json absent. Verify source/input/model hashes, configured selected
steps, 5strictreloads, then copy ONLY aggregate manifest/preflight/completion and
verification/result records locally under results/all_members_ssl_20260915.
No image, label, OOF, checkpoint or model archive download for this wave yet.
Report completed model paths and no unbiased accuracy score. Packaging or
validation-driven promotion requires a separate decision, not automatic action.

Continue existing clear-ec-ssl heartbeat using OpenAI Docs scheduled-task guidance.
Release delivery is COMPLETE and must not be repeated. Duration audit remains
independent and should finish normally. New all-five SSL request takes priority
over queued distillation: do NOT launch distill GPU jobs until these two new
queues are terminal as well as the previously required duration/release gates.
Do not use these all-data weights as internal-fold distillation teachers. Existing
distillation limits/gates and no-new-GPU-afterSept15 10UTC remain; if there is no
time left for its authorized launches, report deadline rather than broaden scope.
Pause monitor only when all branches are terminal; absolute cutoffSept16 00UTC.
Unchanged increments stay quiet; report failure, real phase transition, completion
or needed user action. Helpers areGPU-free and time-limited; remove only ownhelper.

Official automation reference: https://learn.chatgpt.com/docs/automations?surface=app

## Launch verified

02:42:04UTC both initial GPU smokes completed, including strict reload and
four-image40view finite-positive checks. Both queues transitioned to full fitting
(predbag2_35 and metric35). Pods clear-ec-allssl-q0-0915-wrwmk and
clear-ec-allssl-q1-0915-ss9gm wereRunning/restart0 at02:42:50UTC.
No new architecture, raw-label conversion, augmentation or calibration was added.
Existing clear-ec-ssl heartbeat updated in place after official OpenAI Docs review:
same30minute schedule andnotification preferences; now includes these two queues,
marks delivered release complete, and delays distill GPU starts until this wave
is terminal. Distill no-new-GPU deadline10UTC remains unchanged.
