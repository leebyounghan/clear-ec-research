> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../STATRES_SCREEN_20260908.md) for the release-facing summary.

# Statistics-residual matched screen — complete, no improvement

## Final result recovered September9 12:37UTC

Training finished September8 22:47:51UTC; CPU audit and controller finished
22:47:56UTC, done=[0,1],failed={},audit_completed=true. All four branches ran
1490updates and completed full1791image40view evaluation. No new training,
confirmation,all-data fit,candidate promotion,packaging,or submission followed.

On the SAME3582historically reused validation images (lower is better):

| Predictor | CD | CV | HEX | Overall |
| --- | ---: | ---: | ---: | ---: |
| Current6 | 6.351253 | 10.347339 | 10.229259 | 8.975950 |
| Mean residual standalone | 6.880637 | 10.536135 | 10.418161 | — |
| Std residual standalone | 6.885467 | 10.538138 | 10.419246 | — |
| Fixed CV/HEX50/50mean blend, CD unchanged | 6.351253 | 10.381960 | 10.281623 | 9.004945 |
| Fixed CV/HEX50/50std blend, CD unchanged | 6.351253 | 10.384102 | 10.281922 | 9.005759 |

Std minus capacity-matched mean control: CV+0.002003,HEX+0.001085,
mean(CV,HEX)+0.001544points (worse). That target mean worsens in bothfolds:
fold0+0.003034,fold1+0.000054. Std fixedblend vs current6overall worsens
aggregate+0.029809,fold0+0.044643,fold1+0.014974. The screen gate fails.
No statistically significant difference is claimed; the changes vs mean are
tiny and there is no evidence of benefit from THIS residual-std recipe.
This does not prove all distribution-aware models ineffective or diagnose
mean pooling as the causal bottleneck. Do not extend this failed queue.
The8.975950reference is a TWO-FOLD subset, not fivefold9.016114304 or a
challenge score. No calibration or post-hoc blend-weight search was performed.

Actual queue19:52:57–22:47:56UTC =2h54m59s wall,5.832778allocatedGPUh during
queue. Training4.978261GPUh + evaluation0.812257GPUh =5.790518GPUh.

### Delayed monitoring and resource accounting

Last live read September8 22:44UTC found final evaluation1300/1200of1791.
Next actual execution of the delayed heartbeat was September9 12:34UTC;
the supplied trigger time was September8 23:08:36UTC. Cause of the local
execution gap is not established. Do not claim prompt post-completion release.
The independent6hpod safety limit fired: podFailed/DeadlineExceeded at
September9 01:47:49UTC, researcherterminated01:48:20UTC (exit137), rather
than the monitor deleting the pod immediately after work ended.
The container ran19:47:53–01:48:20, ~6h00m27s of2GPUreservation (~12.015GPUh,
NOT measured billing). About3h00m24s elapsed AFTER controller completion
before container termination, ~6.013GPUh of post-work reservation. The
training did NOT fail: completion/audit predate the pod deadline by3hours.

No GPUs were reallocated for result recovery. Temporary`clear-ec-statread-cpu`
requested1CPU/2Gi and ZERO GPUs, with both PVC and volume mount read-only,
NVIDIA_VISIBLE_DEVICES=void and a15minute safety limit. It only read results
and verified source/checkpoint hashes; it did not rerun the audit or training.
Lustre mount confirmedro on existingPVC`clear-ec-workspace`, volume
`pvc-exa-abfcdbc5-269d-4f5c-a069-6716e1606c6a`. Experiment9.1GiB remains there.
All19source hashes and all4fullcheckpoint hashes matched. Each checkpoint
is1,218,834,904bytes; all4OOF files are present.

Local audit copy:`STATRES_SCREEN_20260908_audit.json`.
SHA256:`5dc7298ba9f2a7b9a570a93ecf07d5273fb819b0ea6cf2d1145ad447dc7da86c`.
The original complete.json/audit.log and checkpoints remain onPVC. Existing
candidate files and raw data were not modified or deleted.
AtSeptember9 12:41UTC both`clear-ec-statres-2gpu`and`clear-ec-statread-cpu`
were gracefully deleted after recovery/verification. Deletion wait completed;
subsequent app=clear-ec pod listing was empty andPVC remainedBound on the
same volume. Only these two pod objects were removed, not model/data files;
artifacts remain accessible by remounting the preservedPVC. App heartbeat
`clear-ec-2gpu` was deleted using the app tool with OpenAI Docs guidance.

## Historical experiment design and launch (not current status)

## Scope and rationale

User approved further improvement after the failed specialist screen and limits
GPU usage to2or4. We explicitly interpreted the direction as preserving feature
distribution information, not another identical single-target continuation.
Exactly2H200 allocated to new pod`clear-ec-statres-2gpu`, namespace
`p-ajou-mlxp-project`, container`researcher`, node`h200-04-w-11f1`.
Other workloads and old staged candidates are out of scope.

Mean+std pooling is NOT a new untried idea in this repo: `train_stdpool.py`
already tried concatenation on random image splits, and earlier feature MIL /
global attention also existed. That old trainer excludes its LayerNorm from
its optimizer; this does not by itself explain its performance. New optimizer
coverage includes all trainable normalization and residual parameters.

This new experiment differs: retain the trained mean prediction path and add
a zero-initialized residual forCV/HEX. Compare a spatial-standard-deviation
descriptor against a mean-descriptor residual of EXACTLY equal capacity.
Both residuals add33,666parameters and no extra encoder pass. Standard deviation
is over spatial encoder features, NOT directly over cell areas; the representation
hypothesis is unproven. No new image annotation, attention, JEPA or DINO run.

## Fixed design

- Two historical slide-group folds0/1,1791validation images each,8954population.
- Reuse ONLY `experiments/specialize_screen_20260908/runs/fold{0,1}/common/selected.pt`:
  original CPSAM ->2980train-only updates; no validation checkpoint selection.
  Verify common hash,fold,original initialization,2980updates,non-smoke config,
  train/validation split and official targets. NEVER use all-data weights.
- Both branches independently restore that exact common checkpoint; residual
  initialization and training crop/dropout RNG match across branches.
- Original mean path includes the existing norm and3heads. Residual is separate
  LayerNorm256 -> Linear256x128 -> GELU -> Linear128x2; final layer is zero.
  Residual outputs onlyCV/HEX. The full shared encoder and original heads still
  train; therefore standaloneCD can change. Deployment blend holds existingCD
  exactly unchanged and combines ONLYCV/HEX.
- `mean` control uses float32 spatial mean; `std` uses float32 population variance
  withsqrt(var+1e-6). Added capacity and optimization budget are identical.
- Each branch1490updates (10*149), freshAdamW,encoderLR3e-6,head/normLR3e-4,
  weightdecay1e-4,cosine/warmup,clip1,BF16,activationcheckpoint,BS48,K2,
  full_quality native256 crops from raw972x1296 images, mixedMAPE.
- Same train-only slide smoothing and fixed legacy target normalizers as prior
  screen. Training seed377,residual seed277,common seed77.
- Save terminal checkpoint BEFORE evaluation;1791images*40views/fold/branch.
  No best-epoch selection,calibration,blend-weight search,or automatic promotion.
- Four full jobs total,2workers concurrently eachserial mean->std. Smoke first:
 2updates/branch +4validation images. Smoke scores are not performance.

## Gate and audit

Compare std vs equal-capacity mean control on mean(CV_MAPE,HEX_MAPE): both folds
must improve and aggregate improvement>=0.03percentage points. Fixed blend
uses50%current6+50%new forCV/HEX, originalcurrent6CD untouched. It must improve
current6overall in both folds and aggregate>=0.01points, and beat the mean-control
blend in BOTH folds. No per-target cherry-picking. If it passes, report bounded
remaining-fold confirmation plan only; do not auto-expand this queue.

The controller runs `source/audit_statres_screen.py` once onCPU after bothfolds
complete. It checks IDs,indices,targets,split equality,positive finite predictions,
historicalbaseline hashes,source hashes,all4newcheckpoint hashes and reports
bothstandalone andfixedblend metrics. `complete.json` sets audit_in_progress
duringCPUaudit; do not race it or rerun afterfailure. It contains audit_completed
or audit_error when the controller is terminal. The audit records the input
completion-file hash BEFORE the controller appends final audit status.

Reference current6 on this SAME3582image subset is8.975950250. This is NOT its
full5fold score9.016114304, nor a challenge score. Folds were historically reused,
so passing is exploratory evidence requiring confirmation, not proof of8.x.

## Resources and safety

Total new updates5960; prior measured~3s/update =>4.97GPUh training.
Four40viewevaluations at~730s =>0.81GPUh; total~5.8GPUh,~2.9hwall on2GPUs
plusstartup/checkpoint I/O. Plan3–3.5hwall and refine from actual full progress.
No cloud price known. Previous shared training is reused, not charged again.

Fullworker bound4.5h,smoke30min,noautoretry,absolute controller deadline
September9 01:00UTC. Pod activeDeadlineSeconds21600 (6h) is an independent
resource safety cap if local app monitoring is unavailable. Normal earlier
completion should be followed by audit, artifact verification and graceful
deletion of ONLYthis idle2GPUpod. Do not delete PVC/data/models or otherpods.

Remote repo `/workspace/clear-ec-research`, Python`/workspace/venv/bin/python`.
Experiment`experiments/statres_screen_20260908`;source immutable aftermanifest.
Controller outputs`status.json`,`complete.json`,`screen_audit.json`,logs;
worker outputs`runs/fold*/{mean,std}/{selected.pt,oof.npz,result.json,progress.json}`.
Mount is existing`clear-ec-workspace`PVC; preserve all artifacts beforeGPUrelease.

Four CPU unit tests passed: zero-residual exact prediction equivalence/equal
capacity, same-mean different-variance distinction, finite constant-map gradient,
and residual changes onlyCV/HEX. Python compilation passed. Mac upload metadata
files were moved to`upload_metadata/` before freeze; no training/data was deleted.

## Live launch verification —19:54UTC

ControllerPID430. Smoke workers498/504 both completed allbranches; full workers
542(fold0/GPU0) and548(fold1/GPU1) launched~19:53:30UTC. At19:54:47UTC,
failed={},bothfullworkers/controller alive, sampledGPU96–100%,51771MiB each.
Both GPU smokes verified exact initial prediction preservation, nonzero learned
residual weights, matched crop/RNG hashes and finite positive4image outputs.
The four CPU unit tests and compilation passed beforecontrollerlaunch.
At19:55:07UTC bothfullmeanbranches reached25/1490updates at3.239–3.247s/update,
peak38.309GiB. Status19:55:37UTC remainedfailed={}. All19recordedsource hashes
and the CPUaudit import were verified without modifying running source.

App heartbeat`clear-ec-2gpu` was freshly created with name
`CLEAR-EC 2GPU 분산보정 실험 점검`; checks every10minutes, quiet for normal
progress, reports meaningful completion/failure and releases ONLYthis idlepod
after audit+artifact preservation. This is a new monitor, not the deleted
specialist experiment monitor. OpenAI Docs guidance was used for its setup.

Initial full-queue estimate~22:50–23:25UTC Sep8 (~07:50–08:25KST Sep9 /
18:50–19:25NewYork Sep8), subject to actual measured1490updatebranch progress.
Remaining-fold confirmation is not queued. Existing candidates and all-data
deployment artifacts are unchanged; no challenge submission or Docker work.
