# Phase II priority 1: target averaging audit and matched ablation

## Launch verified — September12 00:59UTC /09:59KST

Job `clear-ec-target-screen-0912`, pod `clear-ec-target-screen-0912-nkbwt`,
node `h200-04-w-11f1`; four H200 assigned. Original/SSL raw-target full workers
on folds0/1 launched after all8 smoke workers passed. Synthetic tests5/5 passed;
smoothed original-CPSAM smoke outputs reproduced prior results within1e-5
relative tolerance. All4 raw/smoothed smoke pairs had identical input sampling
hashes. No full results available at launch, no failures or restarts observed.

Full raw fits started~00:59UTC. Prior throughput predicts roughly05:30-06:30UTC
(14:30-15:30KST), not a guaranteed finish. Job hard deadline~08:58UTC. Native
Istio sidecar confirmed. Source and split manifest locally recovered and SHA
verified under `results/target_screen_20260912/`. No data/OOF arrays exported.

CPU preflight rehashed all8954 images, verified protected controls and all-six
historical split alignment, and passed exact-duplicate-connected group checks.
Its resource use is separate from the4GPU training Job. Original candidate and
all previous experiments remain unchanged.

User asked to pursue the previously proposed priorities, preserving the current
Phase I candidate and focusing on promising Phase II directions. No new submission
is authorized. Prior request allows maximum useful GPU parallelism; this finite
screen requests four H200 and does not modify other workloads.

## Why this bounded experiment

Sep11 SSL screen finished successfully, but its fixed-six replacement only
improved0.00444 points, below its frozen0.02 gate. Standalone SSL improved0.05366
points. These are historically reused two-fold internal results, not leaderboard
scores. It is not proof that all SSL or all ensembles fail.

Target averaging was explored previously: the Sep4 CENTRAL single-crop raw-target
five-fold score9.203243 vs smoothed9.209081 was only a tiny gain. We do NOT claim
raw targets are a new untested idea. This screen tests the CURRENT full_quality
two-crop mixed-MAPE35 recipe, clean duplicate-connected splits, and interaction
with the NEW fold-specific SSL initialization. No crop or loss change is added.

CPU training-only inspection before any new score:
- Fold0:7151 rows,5409 slides;3480 rows changed by slide averaging. Mean absolute
  relative target shift CD/CV/HEX1.786/3.194/3.354%;95th7.243/13.636/14.141%.
- Fold1:7157 rows,5411 slides;3485 changed. Means1.826/3.235/3.405%;
  95th7.247/13.793/14.423%.
These are changes to labels, NOT prediction errors or proof that averaging is bad.
Both denoising and erasure of real within-slide variation are plausible.

## Design fixed before GPU training

2x2 encoder initialization x target policy, each on folds0/1:

| Initialization | Slide averaged | Raw image-level targets |
|---|---|---|
| Original CPSAM | Reuse clean35 from Sep11 | Two new fits |
| Same-fold6000-step SSL encoder | Reuse SSL+35 from Sep11 | Two new fits |

Only target policy changes in each matched pair. All four new fits: seed77,
native256 full_qualityK2, batch48, mixedMAPE, encoderLR1e-5/head1e-3,
fixed terminal35epochs, BF16, same optimizer/warmup/cosine/weight decay,40views
at evaluation. Fold0 has5180updates, fold1 has5215. Fresh heads, no all-data
checkpoint initialization, no additional SSL training. Verify input sampling
hashes match controls for the first3 training minibatches.

All original inputs and controls are hash-verified. Rehash all8954 raw image
arrays against the earlier fingerprints. Slide+exact-duplicate connected groups
must not cross cleaned train/validation. Historical6 alignment checks include
all members' training indices. Primary comparison uses the SAME3568 validation
rows as the SSL screen, excluding historical duplicate-linked groups. Full3582
rows are evaluated for descriptive reporting only. Patient identity and near-
duplicate coverage remain unresolved; this is not untouched/nested validation.

## Fixed decision rules

For each initialization separately:
- Research signal: raw targets beat MATCHED smoothed standalone on both folds
  and by>=0.02 percentage points overall. This is screening, not significance.
- Ensemble confirmation eligibility: fixed replacement `(5*old5+raw)/6` beats
  current6 on BOTH folds and by>=0.02 overall, and beats matched smoothed
  replacement on BOTH folds.
- Report CD/CV/HEX and both folds, standalone and replacement metrics, plus
  difference-of-differences for SSL benefit with versus without smoothing.
- Two prespecified hypotheses; no fitted weights/calibration, no changing gates
  after results. No automatic confirmation, all-data fit, packaging or submission.

## Safety and timing

Synthetic unit tests before allocation. On GPUs run8 short smokes (both policies
for each initialization/fold) before any full fit; verify raw/smoothed sampling
parity and base smoothed numerical reproduction of previous smoke predictions.
No retries. Stop all workers on failure. Four full fits run concurrently.
Each takes about4.3h training plus12min evaluation at recently measured~3s/update;
allow4.5-5.5h including startup/smokes/I/O. Approximately18 GPU-hours of useful
work, depending on actual throughput;8h Job deadline caps request at32GPU-hours.
Controller deadline7h, worker deadline6h, audit timeout30min. Main process exits
after audit, native sidecar stops, completed Job/pod TTL600s. PVC is preserved.

## Next priority, not launched by this queue

Use this result to decide the fixed target policy, then test native-scale region
coverage/selection with matched controls. Previously failed resized-global and
mean/std MIL designs are not to be repeated as if untested. Annotation-based
cell-structure training and a different backbone come later, with provenance and
challenge-use eligibility checked. This queue makes no new external downloads.
