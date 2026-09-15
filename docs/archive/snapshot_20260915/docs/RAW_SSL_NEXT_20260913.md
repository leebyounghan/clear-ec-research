> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../RAW_SSL_NEXT_20260913.md) for the release-facing summary.

# Raw-target release and SSL optimization — September 13, 2026

## Complete — September 13, 12:12:34 UTC

All three fold workers and six regression fits completed successfully. No
failures or interruptions were recorded. The queue ran 39,947.59 seconds
(11h5m48s). At the final readback, source/input hashes and all 24 audit-bound
checkpoint/prediction/config/result hashes were reverified on Kubernetes.
The original three-GPU Job expired after completion; no CLEAR-EC GPU Pods remain.

Identical primary validation cohort: 5,350 images across folds 2/3/4. Lower
uncalibrated mean MAPE is better. These are local validation results, NOT
official scores or a prediction of the leaderboard score.

| Sixth-member method | Standalone | Fixed-six ensemble | Ensemble delta vs raw |
|---|---:|---:|---:|
| Raw labels, original CPSAM initialization | 9.271957 | 9.029185 | — |
| SSL + raw labels, encoder LR1e-5 | 9.222232 | 9.019599 | -0.009586 |
| SSL + raw labels, encoder LR5e-6 | 9.229801 | 9.021819 | -0.007366 |

The standard SSL rate beat raw initialization on all three folds but the
ensemble gain was small. Half LR lost to the standard rate on folds2/3 and in
aggregate; its prespecified research gate is false. No model was automatically
promoted or submitted. Keep the already delivered raw-label all-data archive
unchanged. Standard-rate SSL is the stronger research direction if a future
all-data SSL release is explicitly requested; no all-data SSL model was fitted
or packaged by this experiment.

Measured SSL-stage compute: 6.010 GPU-hours; six regression fits: 25.845
GPU-hours; six evaluations: 1.221 GPU-hours; sum: 33.077 GPU-hours. Three-GPU
controller allocation: approximately 33.290 GPU-hours including overhead.
These are runtime calculations, not monetary billing totals.

Aggregate audit and completion marker are locally preserved in
`results/raw_ssl_next_20260913/`; no patient images, labels or OOF arrays were
downloaded. Final artifact delivery and SSL result reporting satisfy this
heartbeat's stop condition. Do not resume experiments automatically.

## Verified delivery milestone — 06:34 UTC

Local delivery is complete. The resumable downloader exited successfully after
~752 seconds of range transfer, assembled the archive, verified the complete
SHA256, and wrote `download_verified.json` at approximately 06:04 UTC. Final
file: `submission_candidate_raw_20260913/model_full_quality77_raw_20260913.tar.gz`.
Only the downloader's temporary range fragments were removed after success;
the final archive, source archive, original submission and transfer plan remain.
Do not redownload or overwrite this verified file on later heartbeat checks.

At 06:34 UTC, all three SSL standard-LR fits were around 4,150 updates, with no
reported failures; half-LR fits and the final accuracy audit remain pending.
Continue the existing heartbeat for SSL results only; do not repeat the artifact
delivery notification unless its state changes.

Raw all-data training completed all 5,215 updates in 15,255 seconds (~4h14m).
Strict reload, eight-image numerical/interface checks and the frozen application
entrypoint passed. The first five predictions matched exactly; original member
hashes and calibration were preserved. Runtime finished 05:27:01 UTC; CPU archive
creation and all member hash checks finished 05:33:53 UTC. The raw GPU and archive
Jobs subsequently expired normally under their completion TTL. One GPU was freed.

Remote archive is 7,964,624,301 bytes. SHA256:
`5bcf45dc1f057ef431ce9504a31b6506e671e332bb9c0d698644bf662370cec3`.
Local delivery verification is tracked separately in
`submission_candidate_raw_20260913/download_verified.json`; until that exists,
any `.partial` download is not a completed deliverable.

Local transfer started 05:49 UTC. The single-stream copy was deliberately stopped
with its ~990 MB prefix intact, then `code/fetch_raw_release_20260913.py` continued
the remaining ranges with eight download connections. This is downloading and
checksum verification only, not local model computation. It preserves resumable
range files under `submission_candidate_raw_20260913/transfer_ranges/`, appends
only complete ranges, verifies the entire SHA256, then renames `.partial` to the
final archive and writes `download_verified.json`. It removes only its own range
fragments after successful verification. Execution session at launch: 11132.
Future checks: first check final file/verification marker and whether this exact
downloader process is still alive. Do not launch duplicate writers or overwrite
the partial file. If it exited early, its range plan permits resuming the same
script after confirming the named source Pod still exists. Metadata and the
remote readiness marker are already local; neither alone proves download success.

SSL pretraining completed all 6,000 updates in each of folds 2/3/4 at approximately
03:08 UTC. Training-only representation diagnostics passed. Standard-LR raw
regression was around 3,225/5,215 updates (fold4 total5,180) at 05:49 UTC; half-LR
fits and the terminal comparison are still pending. No validation accuracy claim
can be made from training loss or representation diagnostics.

## Historical launch record

User requested both tasks. GPU jobs launched at approximately 01:06 UTC
(September 12, 21:06 EDT / September 13, 10:06 KST). This is a launch record,
not a claim that training or packaging has completed.

At approximately 01:10 UTC all four GPU smoke workers passed and full runs were
active. Raw first 25/5,215 updates: 2.953 s/update. SSL first 50/6,000 updates:
1.217–1.227 s/update across three folds. GPU snapshot: 100% raw and 94/99/99%
SSL, not a sustained-utilization guarantee. Both GPU Jobs and the CPU archive
Job were Running. No completed-model accuracy or ready archive exists yet.

Thread heartbeat `clear-ec-ssl` checks every 30 minutes, delivers a verified
archive when ready and reports terminal SSL results. It does not launch new
experiments or submit models, and stops by September 14 02:00 UTC at the latest.

| Track | Change from the September 10 submission | Resources | Deliverable |
|---|---|---|---|
| Raw-label release | Preserve first five models, equal weighting, calibration and inference interface. Refit sixth full-quality seed77 member on original, unsmoothed labels. | 1 H200 | Separate model archive; existing image reused |
| SSL + raw optimization | Fold-specific 6,000-step native-scale SSL, then compare encoder LR 1e-5 vs 5e-6; heads use 100× encoder LR. Raw regression labels in both. | 3 H200, folds 2/3/4 in parallel | Six downstream fits, OOF comparison, research gate |
| Packaging | Hash-check eight flat-root files and read back the compressed archive. | CPU only | tar.gz, SHA256SUMS.txt, model_archive.json |

The new release is NOT an existing validation-fold checkpoint. It is a fresh
all-data fit on 8,998 eligible images, 5,215 optimizer updates, batch 48, two
native crops per image. The fixed update count is the median terminal raw CV
count, not a validation-selected checkpoint or 35 all-data epochs.

The SSL trial uses the existing custom masked EMA feature objective with online
variance/covariance regularization, not an official I-JEPA implementation. Each
fold excludes its validation images and duplicate-linked training slide groups
from pretraining and regression. No scale-changing resize, external corpus,
new architecture, calibration fitting or automatic SSL promotion is added.

Five CPU contract tests and the complete data/source preflight passed before
launch. Each GPU worker must pass a separate short fit/reload/inference smoke
test before its full run. Code and inputs are hash-frozen on the PVC.

## Technical release gate

After raw all-data training, preserve the existing five member hashes and replace
only `full_quality_seed77.pt`. Verify strict checkpoint reload, eight-image FP16
and BF16 inference, unchanged first-five predictions, unchanged aggregation and
calibration, and the three-output application entrypoint. The runtime is a fresh
isolated CUDA 12.6 / PyTorch 2.11 environment with the frozen submission app.
This is an H200 interface/numerical test, NOT an actual archived Docker image or
official T4 test. Official platform validation remains pending.

Only successful runtime checks unlock CPU archive creation. Archive name:
`model_full_quality77_raw_20260913.tar.gz`. It contains the six model checkpoints,
foundation weight and original configuration. No patient images, labels or OOF
arrays belong in the archive. The September 9/10 submission files are unchanged.

## Result interpretation

Prior five-fold raw vs newly matched smoothed-label comparison improved fixed-six
uncalibrated mean MAPE by approximately 0.00935. This is modest and does not prove
an official score improvement. Previously specified 0.02 magnitude gate was not
met. Packaging is now explicitly user-requested, not automatic score promotion.

For the new LR comparison, report every variant on identical primary validation
rows. Half LR qualifies for further research only if it beats both standard SSL
LR and the original-initialization raw control in every tested fold and by at
least 0.01 in aggregate fixed-six MAPE. No candidate is submitted automatically.

## Timing and limits

Historical rates: SSL approximately 1.2 s/update, regression approximately
3.0–3.2 s/update. Raw fit approximately 4.4–4.7 h plus startup/runtime/archive;
provisional model archive estimate 5–6 h after launch. Each SSL worker has about
2 h pretraining + two approximately 4.5 h regression fits + evaluation overhead;
provisional total 11–13 h. These are estimates, not completion guarantees.

Jobs have no retries. Raw controller is bounded to 7 h plus post-training checks,
outer runtime 7.5 h, Job deadline 8 h. SSL controller 13 h, outer runtime 13 h20m,
Job deadline 14 h. CPU archive waits at most 11 h, Job deadline 12 h. Finished
Jobs expire after 10 minutes; all result files persist on the existing PVC.

## Locations

- PVC `clear-ec-workspace`, namespace `p-ajou-mlxp-project`.
- Remote root: `/workspace/clear-ec-research/experiments/raw_ssl_next_20260913/`.
- Immutable source and manifest: `source/next_candidate_20260913.py`, `manifest.json`.
- Raw: `raw_status.json`, `raw_logs/`, `raw_deploy/`, `raw_complete.json`.
- SSL: `ssl_status.json`, `ssl_logs/`, `ssl_runs/`, `ssl_complete.json`, `ssl_audit.json`.
- Package: `package_ready.json`, `artifacts/`; these do not exist until successful completion.
- Intended local delivery: `submission_candidate_raw_20260913/`; no archive yet at launch.
- Jobs: `clear-ec-raw-release-0913`, `clear-ec-ssl-raw-opt-0913`, `clear-ec-raw-archive-0913`.

The CPU archive Job was initially rejected before creating a Pod because its
zone selector was absent. That empty Job was recreated with the same permitted
private-zone selector as the preflight Pod; no experiment data was deleted.

Local Mac is used only for orchestration, source editing and eventual download.
Kubernetes training and archive creation continue independently of the Mac.
App-based follow-up and local downloading require the Mac/app to be available.
