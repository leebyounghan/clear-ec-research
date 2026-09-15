# Historical training reproduction notes

These recipes describe the September 10 baseline. The first five members are
retained in the September 14 release, but its sixth member instead follows the
[SSL/raw release recipe](../docs/SSL_RELEASE_20260914.md). For reviewer inference,
use the [Docker guide](../docker_review_20260914/README.md); retraining is separate.

This is a faithful source/recipe inventory, not a claim that clean retraining has
already passed. The submitted weights are the reproduction target; inference from
those weights is the primary scored-result reproduction path.

## Selected fits

| Checkpoint | Seed | Native crop source | Crops/train sample | Loss | Selected updates |
|---|---:|---|---:|---|---:|
| seed42.pt | 42 | Central 512 field | 1 | Standardized log L1 | 3,576 |
| seed77.pt | 77 | Central 512 field | 1 | Standardized log L1 | 2,831 |
| seed2026.pt | 2026 | Central 512 field | 1 | Standardized log L1 | 2,980 |
| metric35.pt | 42 | Central 512 field | 1 | Mixed log L1 / relative error | 4,470 |
| predbag2_35.pt | 42 | Central 512 field | 2 | Standardized log L1 | 3,874 |
| full_quality_seed77.pt | 77 | Quality-selected full native field | 2 | Mixed log L1 / relative error | 4,917 |

All use batch size 48 with partial final batches, a Cellpose-SAM encoder,
AdamW (encoder LR 1e-5, head/norm LR 1e-3, weight decay 1e-4), BF16 autocast,
activation checkpointing, gradient clipping at 1 and fixed random seeds.
The first three runs completed 4,700 updates/25 epochs, but the submitted weights
are the earlier `selected.pt` snapshots, not the last epoch.
The first three use 200-update warmup then constant LR. The other three use
200-update warmup and cosine decay over 5,215 nominal updates with a 0.05 floor.

The historical first-five `training/code/` sources differ from the regional
frozen source under `training/experiments/region_search_20260905/source/code/`.
In particular, the old two-crop builder constructs a replacement prediction-bag
head after the ordinary model; the regional builder constructs the head once.
This changes RNG consumption. Do not modernize the builder and call it an exact
historical rerun. SHA256 checks bind both versions to the recorded fit configurations.

## Data and targets

The provided raw training archive has 9,000 images. The old exact-ID intersection
contains 8,954 images. Preserve those matches, then reconcile the remaining 46
separator-only ID aliases with one-to-one checks. Exclude the two rows with a
nonpositive metric before slide averaging, giving 8,998 eligible rows. Average
positive labels within each slide for this historical all-data fit.

No official validation/test images should be used to recreate these fits. The
selected implementation does not require the train_green ROI annotations or any
new domain-SSL checkpoint. The public foundation model remains a required resource.

## Preparation sequence (requires clean-data validation before release)

The following commands are run from `training/`. First obtain the authorized
raw archive containing `train_mha/*.mha` and `final_train_ids.csv`. Obtain these
from the official challenge distribution or an organizer-authorized copy; do
not distribute the training data with this repository. Record its SHA256.

Bootstrap `datasets/clear_ec_official/` with the original, unmodified CSV and
raw `train_mha/` directory. Do not normalize CSV IDs at this first step: the
historical exact-ID cache must have 8,954 rows in sorted filename order.
The bootstrap acquisition/extraction procedure and archive hash still need to
be validated against the official reviewer-accessible release.

```bash
cd training
python code/cache_native.py
python code/prepare_fullframe_cache.py \
  --archive /absolute/path/to/authorized/raw-archive.tar.gz \
  --sha256 "REPLACE_WITH_VERIFIED_RAW_ARCHIVE_SHA256"
python code/audit_raw_id_aliases.py
python code/prepare_final9000_candidate.py
python experiments/region_search_20260905/source/code/fullframe_views.py
```

`cache_native.py` overwrites its cache files: execute only in a new training
workspace, never over an existing research dataset. Later preparation stages
verify row/label/central-pixel alignment and stop on inconsistencies.
The global 256-pixel cache produced by the final command is an inherited loader
dependency; the submitted full_quality model uses native local tiles, not that
resized global image as an input view.

Place the verified foundation file at `training/cpsam_models/cpsam_v2`.
Expected generated resources:

- `datasets/clear_ec_official/{img_native_512.dat,cache_native_meta_512.npz}`
- `datasets/clear_ec_fullframe_20260904/{train_mha/,source_metadata/,img_full_raw.dat,fullframe_meta.npz,fullframe_manifest.json,context_views_manifest.json,context_quality.npz,img_global_256.dat}`
- `datasets/clear_ec_final9000_20260904/{img_native_512.dat,meta.npz,manifest.json}`

## All-data fitting commands

Run in a **separate clean training environment**, not inside the read-only inference
container. Historical training used PyTorch 2.11 / CUDA 13 on H200. The exact
historical training Python, transitive dependency and base-image lock is still
to be recovered and smoke-tested; the inference CUDA 12.6 lock is not a claim
of bitwise training equivalence. Use one H200 per process. Do not assume the
batch-48 BF16 recipe fits or works unchanged on a T4.

From `training/`, with no previous run outputs:

```bash
CUDA_VISIBLE_DEVICES=0 python code/train_final_candidate.py --seed 42
CUDA_VISIBLE_DEVICES=0 python code/train_final_candidate.py --seed 77
CUDA_VISIBLE_DEVICES=0 python code/train_final_candidate.py --seed 2026
CUDA_VISIBLE_DEVICES=0 python code/train_deploy_methods.py --recipe metric35
CUDA_VISIBLE_DEVICES=0 python code/train_deploy_methods.py --recipe predbag2_35
CUDA_VISIBLE_DEVICES=0 python experiments/deploy_region_20260907/source/train_deploy_region.py \
  --recipe full_quality --output-root experiments/deploy_region_20260907
```

Outputs are `selected.pt`, `config.json`, progress and result records beneath:

- `experiments/final_candidate_20260904/runs/seed{42,77,2026}/`
- `experiments/deploy_methods_20260905/runs/{metric35,predbag2_35}/`
- `experiments/deploy_region_20260907/runs/full_quality/`

Historical elapsed fitting times were approximately 1.83, 1.84, 1.83, 1.73,
3.23 and 4.01 hours respectively: about 14.46 summed single-GPU process-hours.
These exclude preparation, cross-validation search and evaluation, and are not
allocated/billed GPU-hours. New hardware/software and I/O may differ.

Never overwrite the submitted checkpoints with rerun outputs. Keep the six-member
calibration frozen. Refit outputs need their own identity and numerical audit.

## Remaining provenance gates

The original three-seed selection plan is preserved verbatim; its calibration
and three-member weights are historical selection metadata, **not** the six-member
inference configuration. Use only `configs/candidate6_full_quality.json` at inference.
The fit configurations are in `provenance/training/`, outside run folders so they
do not trip interrupted-run protection.

Before calling this a complete research reproduction release, add/review the
historical OOF split generation, selection/calibration scripts and required
non-image evidence; close all source dependencies for those optional analyses.
The current directory preserves the final all-data fitting path, not the entire
search history. Validate a clean data bootstrap, all trainer smokes and inference
parity, and document numerical tolerances. Historical OOF audits later found
duplicate-image overlap, and calibration estimates were exploratory rather than
nested validation; do not describe those figures as unbiased leaderboard estimates.
