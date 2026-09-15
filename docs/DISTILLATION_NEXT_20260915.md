# Not started; deadline closed: fold-matched SSL student distillation

## September 15 post-deadline status

The new-GPU launch cutoff (10:00UTC) has passed. PVC inspection found no
`experiments/distillation_20260915` runtime root and no distillation jobs/pods.
Only the loss-contract preparation below was implemented/tested; no teacher gate,
cache or student training result exists. No late GPU launch was performed.
This follow-up is terminal due to the missed execution window, not an observed
negative distillation result. A new schedule/authorization is needed to resume.
The queued plan below is preserved as history, not a current launch instruction.

Authorized September14: user requested finishing SSL and then trying distillation.
This is a bounded follow-up, not a request to change the pending submission.
All computation/tests/training/cache generation stay on K8s. Local work is source,
orchestration and aggregate results only. No new external datasets/weights required.

## Trigger and current status

QUEUED, NOT TRAINING. A small loss contract is implemented in
code/distillation_core_20260915.py; the cache/training runner and frozen manifest
must be implemented and tested by the continuation BEFORE creating any GPU Job.
Do not represent this plan as already running or as a performance improvement.

First finish/verify experiments/ssl_duration_20260914/comparison.json and both
queue completions, and verify the independent ssl_release_20260914 GPU completion.
Submission download can proceed independently; never wait for distillation to
notify the user of a verified submission archive. Do not alter release sources,
artifacts, inference/calibration or existing training jobs. If either prerequisite
training fails, diagnose/report and stop this follow-up without retries.

Continuation is attached to existing clear-ec-ssl heartbeat in this same thread;
the newest distillation-specific prompt overrides older 'no new GPU' restrictions
ONLY for the exactly bounded stages below, after their prerequisites pass.
No new GPU starts after September15 10:00UTC. Absolute follow-up cutoff is
September16 00:00UTC; report state and pause after all existing tasks are terminal.

## Teacher and student definition

Do NOT use all-data submission weights as teachers for internal fold validation.
Do NOT blindly use the old first-five models: their historical validation-selected
checkpoints/splits are not a new, clean, fixed-terminal teacher experiment.

Use four already-trained, same-fold, fixed-terminal raw-target regressors:

- raw: manifest folds[f].control_root from raw_ssl_next_20260913;
- standard SSL6000: raw_ssl_next_20260913/ssl_runs/fold{f}/standard;
- SSL2000 and SSL4000: ssl_duration_20260914/runs/step{step}_fold{f}.

Teacher predictions: arithmetic equal-weight mean of DECODED CD/CV/HEX predictions,
no calibration or fitted weights. This is a research ensemble of four matched
models, not the six-member deployed ensemble and not a segmentation expert teacher.
All teacher training indices must EXACTLY equal the current fold training indices,
with matching validation indices and raw target policy. Verify no slide/exact-dup
overlap through the frozen split provenance; record patient/near-duplicate limits.
No validation images or labels may enter SSL teachers or student training.
Historically reused folds and method selection are NOT independent validation.

Student remains the SAME single CPSAM CandidateNet architecture, no smaller model.
Select SSL initialization among2000/4000/6000 using fold2 standalone MAPE only,
with deterministic tie preference6000 then4000 then2000. Do not consult fold3
for this selection. Use that step for BOTH folds' fold-specific initializers.
Initialize encoder from the corresponding pretrain/encoder_stepXXXX.pt, fresh
seed77 heads; never initialize from a trained regression student checkpoint.
Its existing matched non-distilled regression is the alpha0 control.

## Stage0: CPU-only evidence/provenance gate

Implement isolated root experiments/distillation_20260915 with exclusive creation,
and a runnable source copy and manifest hash. Preserve all prior sources.
Verify source/input/teacher/student-initializer hashes and completed result records.
Align each teacher OOF ids,indices,targets,split and primary masks exactly.
Report same3572 primary rows and valid CD/CV3572/HEX3570, metric-specific masks.
Compute the fixed four-teacher ensemble using existing OOF arrays on PVC.
Choose student by fold2 only, then require teacher fold2 mean at least0.01 below
that student's fold2 mean before any GPU cache work. Report fold3 separately;
do not use it to retune teacher weights/members. If gate fails, save report, notify
user and stop: do not substitute another teacher or launch a sweep automatically.

This gate does not guarantee that train soft targets generalize: teacher predictions
on its own training rows are in-sample. Raw labels remain authoritative. This first
experiment does not implement nested out-of-fold soft-target generation.

## Stage1: fixed teacher caches, maximum two one-GPU Jobs

After CPU gate and tests pass, freeze sources/manifest BEFORE launch. Names:
clear-ec-distill-cache-f2-0915 and clear-ec-distill-cache-f3-0915.
One H200/6CPU/request24Gi/limit64Gi each, timeout5400s, backoff0,
Job deadline6000s, TTL600. PVC and image/zone from existing duration manifests.
No GPU waits for another Job; each exits immediately after verified cache writes.

Load each teacher separately, strict checkpoint restore, eval/inference mode,
no gradients. Generate teacher outputs on the fold training rows and validation
rows in SEPARATE caches, explicit ids/indices and hashes. Match native full_quality
preprocessing; five deterministic native256 quality views, WITHOUT D8 for caches.
Mean encoded outputs across those views per teacher, decode, then arithmetic
mean across the four teachers. Re-encode ensemble mean with fixed LM/LS.
This reduces cache cost but is a 5-view approximation to the 40-view OOF gate;
must be disclosed and checked. Never average encoded outputs across teachers.
NaN,nonpositive predictions and index leakage stop the run; no epsilon repair.
Store all individual arrays ONLY on PVC. Verify coverage/unique rows/hashes.

After both caches complete, CPU-audit the 5-view teacher validation predictions.
Require the SAME fold2 improvement threshold0.01 against the chosen student's
existing 40-view control. If this gate fails, report and stop, no GPU student fit.
This is a fixed compute/quality guard, not new teacher weight selection.

## Stage2: two matched student fits, maximum two one-GPU Jobs

Names clear-ec-distill-student-f2-0915 and clear-ec-distill-student-f3-0915.
One H200/6CPU/request24Gi/limit64Gi each, timeout21600s, backoff0,
Job deadline22500s, TTL600. Each independent Job runs smoke then fresh full fit.
No extra alpha sweep; a SINGLE fixed distillation weight0.2.

Loss = existing mixed_mape(raw labels) +0.2*SmoothL1(student encoded prediction,
detached encoded teacher mean). Keep original raw-label term weight1.0; never
replace original labels with teacher pseudo-targets. Use tests in the supplied
distillation core. Teacher cache targets are image-level multi-view consistency
targets; they do not identify which student crop is best and may introduce bias.

Preserve frozen shared.train loop, optimizer,schedule,seed77,crop sampling exactly.
One scoped adapter can concatenate z_raw and z_teacher, then split them in the
target_loss hook, leaving shared.train unchanged; verify its loaded implementation
before using that adapter and restore monkey-patched functions after the fit.
Train only exact tr indices; never feed validation teacher cache into optimizer.
Same K2,batch48,lr1e-5/head1e-3,5215updates,native256,40-view evaluation.
Smoke tests: alpha0 loss/gradients equal original; teacher detached; valid-target
mask boundaries; strict reload; finite predictions; matched first-three-batch
hashes. To test the three-batch hash use at least3 smoke updates, then reinitialize
for full training. Freeze and verify exact initial model/target/first-three-batch
hashes against the alpha0 control. Nonfinite or mismatch -> fail, no auto retry.
No validation-best checkpoint selection. Save fixed terminal checkpoint and OOF.

## Stage3: CPU audit and stop

Verify all checkpoint/cache/source/input/split/OOF hashes, IDs, targets and reload.
Report raw,selected SSL student baseline,teacher ensemble40view and5view,distilled
single student: CD/CV/HEX mean and both folds on exactly matched primary samples.
Distilled research gate: pooled mean>=0.03 lower than selected SSL control, and
both folds lower. Report all outcomes, even if gate fails. No significance or
leaderboard gain claim. No new all-data student fit or automatic model replacement,
archive/ensemble/calibration changes, submission, Git push or external upload.
Successful results inform a later user decision. Copy aggregate JSON only locally.

## Compute bounds and timeline

At most TWO research GPUs concurrently, and only AFTER existing SSL/release GPU
work finishes. Maximum four new Jobs: two cache + two student. Runtime budget
2*1.5+2*6=15GPUh; scheduling/termination overhead separate. No automatic retries.
Expected student fit alone:5215*2.97/3600=4.30h plus40view evaluation~0.19h;
two folds parallel~4.5h wall/9GPUh plus cache,smoke,load/audit. Cache throughput
must be measured, not guessed as a guarantee; total first result roughly5–6h
after distillation starts if normal performance. CPU preflight may stop it early.
All jobs release on completion; do not keep GPUs reserved while waiting for audit.

If implementation/provenance cannot satisfy these contracts, report the exact
blocker instead of quietly broadening the experiment or using all-data teachers.

## Preparation completed September14

The distillation loss contract passed four K8s CPU test groups: arithmetic decoded
teacher mean/re-encoding roundtrip, no teacher gradient, exact alpha0 supervised
loss AND gradient parity, and rejection of NaN/zero/negative teacher predictions.
Preparation copy only: experiments/distillation_prep_20260914/source/
distillation_core_20260915.py. This is not the future frozen runtime root.

Existing raw and standardSSL6000 teacher checkpoints were SHA-verified and their
split indices exactly matched the intended folds. Both are raw-target,5215-update,
strict-reload-complete regressors. Fold2 has7153 training/1791 validation rows;
fold3 has7161 training/1791 validation rows. SSL2000/4000 teachers are still being
trained and MUST pass their later checks before any distillation is launched.

Existing clear-ec-ssl heartbeat was updated in place, every30minutes, ACTIVE,
named CLEAR-EC 제출·SSL·증류 후속 실험. It preserves release delivery and duration
audit while queuing the bounded distillation implementation/provenance/cache/
student stages above. No new distillation GPU Job exists yet. No standalone
automation or new thread was created. The temporary GPU-free preparation helper
is removed after these checks; existing training and PVC remain unchanged.
