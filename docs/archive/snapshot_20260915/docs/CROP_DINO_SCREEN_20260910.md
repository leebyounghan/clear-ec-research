> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../CROP_DINO_SCREEN_20260910.md) for the release-facing summary.

# September 10 crop/DINO follow-up — crop screen completed, gate not met

## Final result — September 11

All four fits completed05:54:06UTC, failed={}; fixed audit completed05:54:14UTC.
The researcher container exited0 at05:54:14UTC and native sidecar exited0 at
05:54:15UTC; pod phase Succeeded, zero restarts. Thus the delayed monitoring
recovery around12:25–12:31UTC did not leave an active GPU workload until recovery.

Same historically reused3582validation images from folds0/1, raw MAPE (lower better):

| Comparison | Overall raw MAPE |
| --- | ---: |
| Current six-member baseline |8.975950250|
| Fixed add7, matched original-crop control |8.982685536|
| Fixed add7, artifact_guard_v1 |8.964351549|

Guarded improves current6 by0.011598700points, below the predeclared0.02point
gate. Both folds improve current6 (-0.006329546/-0.016867855) and matched-control
add7 (-0.017299449/-0.019368524). Standalone control9.256331914 versus
guarded9.109864608. This is a positive but insufficient screen, NOT a passed
promotion gate, independent test, significance result or leaderboard score.
No remaining-fold confirmation, all-data fit, DINO, new packaging or submission
was launched. Existing submission candidate is unchanged.

Four fits each completed5215updates and strict checkpoint reload. Measured
training17.565121GPUh, evaluation0.804632GPUh; queue9h13m53s wall on2H200,
approximately18.462895 allocated GPUh during the queue, excluding startup and
the few seconds of final audit. No monetary billing estimate is available.

Recovered only nine aggregate/provenance JSON reports through bounded zero-GPU
pod clear-ec-crop-results-reader with read-only PVC mount (verified ro).
Local copies: results/crop_screen_20260911/, all nine SHA256 match remote bytes.
Audit SHA256:60e4c6644998359a70c6df90d2f7044ddc935709a1abc24d1e3c3cda22fc2161.
No images, labels, embeddings, OOF arrays or checkpoints were copied locally by
this recovery. The original audit verified checkpoint hashes and aligned IDs,
targets and splits. Persistent data/models remain on clear-ec-workspace.

Finished GPU pod and recovery helper graceful deletion both completed, with
successful deletion wait. PVC remained Bound500GiB on original volume.
Heartbeat clear-ec-crop-2gpu deleted through automation tool at~12:31UTC, using
OpenAI Docs for supported scheduled-task handling. Overall research is NOT complete.

---

## Historical launch record

User approved the two-H200 plan; current submitted-candidate files are protected.
No Grand Challenge upload, Try Out, scored submission, retraining of the packaged
candidate, or modifications to original data/checkpoints are authorized by this queue.

## Current work and exact paths

- Namespace: `p-ajou-mlxp-project`.
- GPU pod: `clear-ec-crop-screen-2gpu`, exactly two H200 on one node.
- Temporary CPU preparation pod: `clear-ec-crop-preflight`, zero GPUs, deleted
  after source/provenance copies completed. Original PVC retained.
- Existing PVC: `clear-ec-workspace`,500GiB; NEVER delete it.
- Remote repo: `/workspace/clear-ec-research`.
- Experiment: `experiments/crop_dino_screen_20260910`.
- Python: `/workspace/venv/bin/python`, PyTorch2.11.0+cu130.
- Local root: `/Users/byounghanlee/Documents/kubectl/clear-ec challenge/clear-ec-research`.

Queue launched20:40:12UTC September10 (05:40KST September11). At20:42:23UTC
all four GPU smokes had passed, failed={}, full control fits were active:
fold0PID154/GPU0 and fold1PID160/GPU1. Sampled utilization100%/99%, memory
51,769MiB each. Guarded full fits are queued after the respective controls.
No new validation score is available yet. Local manifest copy:
`results/guarded_crop_manifest_20260910.json`.

At20:42:36UTC both control workers reached25/5215updates,3.002–3.014s/update,
finite losses and38.308GiB peak allocated tensor memory (different from the
51,769MiB process footprint shown by nvidia-smi). This is an early throughput
sample, not a final ETA. At20:43:13UTC controller failed={} and both full fits
remained active. Conservative first-screen result target is September11
17:40–19:40KST, plus audit if a timeout reaches the queue cap.

## Preflight completed at20:37:19UTC /05:37:19KST September11

Synthetic unit tests:6/6passed. Initial CPSAM SHA256 matched
`0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667`.
Two slide-group splits and historical OOF IDs/targets were exactly aligned.
The current6 baseline and previously audited input hashes were checked before
GPU allocation. Source hashes are frozen in the new manifest.

Quality audit used ONLY5,370rows in the intersection of fold0 and fold1 training
sets, excluding both validation sets. No target values were used to design or
fit the selection rule. The fixed per-image function was cached independently
for8,954images; this is not a dataset-fitted selector. Recomputed native-image
selections matched cached selections on four training rows.

| Training-only image-quality audit | Original full_quality | artifact_guard_v1 |
| --- | ---: | ---: |
| Selected tiles rejected by fixed quality rule |4.2384%|0.0782%|
| Images with at least one such tile |15.0838%|0.1862%|
| Mean informative-block fraction |93.3475%|97.4358%|
| Mean clipped-pixel fraction |2.1650%|0.4058%|

These are heuristic image-quality measures, NOT ROI/cell ground truth or proof
of prediction improvement. Inference selection/order changes in99.18%of audited
images, so the controlled downstream comparison remains necessary.

Local audit: `results/guarded_crop_preflight_20260910.json`.
Remote selection cache SHA256:
`99893587656a74b5d7992aa8680ff87fb0a0ab490f97f3a34dcc65a480395333`.

## Fixed crop experiment

Original full_quality versus artifact_guard_v1, each folds0/1. One GPU per fold;
within a fold run the control and then guarded policy. All four two-update,
four-image40-view smokes must finish before any full fit.

Each full fit starts from original CPSAM, seed77, native256 pixels, two crops,
batch48,35epoch-equivalent fixed terminal updates, encoderLR1e-5/headLR1e-3,
mixedMAPE, cosine/warmup, BF16, activation checkpointing, clipping1. All trainable
normalizations are included in the optimizer. Slide smoothing is training-only.
No validation checkpoint selection. Each final checkpoint is strictly reloaded
before all1791validation images are evaluated with5positions×D8=40views.

The new policy rejects tiles with excessive saturation or too few locally
informative32-pixel blocks, penalizes their focus score and favors spatial
coverage. Fixed thresholds: at least60%informative blocks, at most20%clipped
pixels. Within a block,5th–95th percentile span>=12, mean12–243, clipped fraction
<35%. These constants were fixed before any new prediction scores. If fewer
than eight acceptable grid tiles exist, use a deterministic least-bad fallback
to preserve the view count. Jitter±32is checked again and falls back to the
selected tile when rejected; no additional RNG draws are introduced.

Shared frozen training functions come from the completed specialist experiment;
this DOES NOT initialize from specialist/common/all-data weights. Those helpers
and the older baseline-audit imports are also covered by source hashes.

## Decision gate and bounds

Fixed add7=(6×current6+new prediction)/7; no fitted blend weights or calibration.
Guarded must improve both folds and aggregate raw MAPE by at least0.02points
versus current6, AND beat the matched-control add7 blend in BOTH folds.
This is an engineering screen, not a significance test. All folds were reused
historically. Passing permits review for remaining-fold confirmation, not
automatic promotion or a leaderboard claim. This queue does NOT launch further fits.

Prior measured K2training526s/epoch gives5.11GPUh per fit,20.46GPUh for four fits,
about10.23hours on two GPUs, plus evaluation/startup/new quality overhead.
Initial planning range12–14hours; update using actual full-run step timings.
Controller cap14h, worker cap7h, smoke cap15min, absolute stopSeptember11
15:00UTC. Pod safety deadline16h. These are resource bounds, not measured billing.
No retries. Queue exits its main container after audit rather than sleep forever.
Native injected sidecar lifecycle and terminal pod release must still be checked;
delete the exact finished/idle pod after preserving reports. Never touch other pods.

## DINO status — eligibility pending, NOT launched

Earlier `train_dino.py` used random image splitting and resized256to224. Its
scores are not reused as slide-group evidence. Planned new probe is official
DINOv2-B/14 frozen features plus a small regression head, then only conditional
fine-tuning. No DINO extraction, head training or new domain pretraining has run.

The official DINOv2 model card lists Apache2.0. CLEAR-EC public homepage and
organizer page confirm September14 PhaseI deadline but do not establish external
pretrained-weight eligibility. The guessed Rules/Participation pages were not
available through web; a normal browser could read the public home page but it
showed no external-weight rule. User was asked for that specific rule text.
Do not claim DINO competition eligibility until confirmed. Do not fabricate ROI
labels, perform synthetic annotation, or send restricted images to external AI.
Existing train_green labels have not been used in this new rule.

Sources checked:

- https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- https://clear-ec.grand-challenge.org/
- https://opticlab.med.upenn.edu/clear-ec-challenge
- https://zenodo.org/records/21270595

## Monitoring and completion handoff

Read `status.json`, `logs/*`, `smoke/{policy}_fold{fold}/result.json`, then
`runs/{policy}_fold{fold}/progress.json` for actual status. Complete markers:
`complete.json`(fits), `screen_audit.json`(comparison), `queue_finished.json`(audit
done). A failed/expired pod alone does not mean training failed; inspect markers.
Preserve aggregate reports locally without copying raw images or embeddings into
conversation. If terminal pod cannot exec, a bounded read-only zero-GPU helper
may recover reports. Remove that helper afterward. Do not reallocate GPUs to read.

Follow-up checks should remain quiet for normal unchanged progress, notify on
completion/failure/deadline risk/user action, never restart or expand this queue,
and remove their own heartbeat after reporting terminal results and cleanup.

Heartbeat `clear-ec-crop-2gpu` created ACTIVE at10-minute intervals, targeting
this task. OpenAI Docs skill used for supported app follow-up setup. Local
follow-up requires this computer and app to remain running; remote training and
controller/pod deadlines do not depend on the app. Documentation:
https://learn.chatgpt.com/docs/automations?surface=app
