# Standalone training entry points — September 14 recipe

This optional path trains new checkpoints. For reproducing submitted predictions,
use the [verified Docker guide](../docker_review_20260914/README.md) or the
[verified source-venv procedure](../docs/legacy_phase1/VENV_SMOKE_20260915.md).

The first-five frozen trainers are preserved unchanged. The sixth-member adapter
calls the original SSL and raw-label regression functions directly, without old
queue manifests, completed research runs, cluster configuration, or old venv paths.
Dated source/output directory names preserve the original trainers' interfaces;
they are created from this repository, not fetched from our research server.

## Environment and explicit inputs

Use Linux x86_64 and a BF16-capable NVIDIA GPU with sufficient memory for the
original batches (SSL 32, regression 48 with two crops). Historical full fits used
H200. The 4090 inference check and tiny-network loop test do not certify that full
training fits a 24 GB GPU. Allow at least 60 GB free disk beyond existing raw inputs
for caches, extracted data, and multiple training checkpoints.

From the repository root, install the tested Python 3.12 / PyTorch 2.11 cu126 venv:

```bash
LAMDA_PYTHON=python3.12 bash review_submission_20260910/scripts/setup_venv.sh
source review_submission_20260910/.venv/bin/activate
python training_reproduction/train_ssl_raw.py --check-imports
python training_reproduction/test_portable.py
```

The original training runtime used CUDA 13; this installed CUDA 12.6 environment
has passed inference and synthetic loop checks, not a complete historical refit.
Do not install `env/requirements-clearec.txt`: it describes a different research
environment, not this execution path.

Obtain authorized challenge data independently; this repository does not distribute
it. Required inputs are an unpacked original `train_mha/` folder (9,000 MHA files),
the original `final_train_ids.csv` (not the reconciled CSV), the corresponding raw
archive, and the released `cpsam_v2`. Record the raw archive's checksum from your
authorized transfer. Its contents must be `train_mha/*.mha` and the original CSV.

## 1. Initialize a new workspace

Replace every example path. Keep the checkout, workspace and released models
separate. This command rejects an existing workspace, verifies the foundation
hash and the 8,954 exact ID intersection, copies source, and links the read-only
input resources. It does not train or create image caches.

```bash
python training_reproduction/workspace.py init \
  --workspace /absolute/path/to/new-training-workspace \
  --raw-dir /absolute/path/to/original/train_mha \
  --labels /absolute/path/to/original/final_train_ids.csv \
  --foundation /absolute/path/to/released/model/cpsam_v2
```

## 2. Prepare data, without old experiment results

```bash
python training_reproduction/workspace.py prepare \
  --workspace /absolute/path/to/new-training-workspace \
  --archive /absolute/path/to/clear-ec-train9000-raw.tar.gz \
  --sha256 "REPLACE_WITH_INDEPENDENTLY_RECORDED_RAW_ARCHIVE_SHA256"
```

The frozen preparation sequence builds central/raw caches, checks source pixel
alignment, reconciles 46 separator-only ID aliases, excludes two nonpositive rows,
creates slide-averaged targets for the first five, retains raw targets for the
sixth, and computes label-free region quality. Logs are `prepare_0.log` through
`prepare_4.log`; successful completion creates `data_ready.json`.

Preparation refuses automatic reruns after a started attempt. If it fails, inspect
the logs and use a new workspace after correcting the cause. Do not delete your
original inputs or overwrite the submitted weights.

## 3. Explicitly launch the first five fits

Activate the venv before changing directory. Each command owns one GPU and runs
sequentially below; there is no job scheduler or automatic background queue.

```bash
cd /absolute/path/to/new-training-workspace
CUDA_VISIBLE_DEVICES=0 python code/train_final_candidate.py --seed 42
CUDA_VISIBLE_DEVICES=0 python code/train_final_candidate.py --seed 77
CUDA_VISIBLE_DEVICES=0 python code/train_final_candidate.py --seed 2026
CUDA_VISIBLE_DEVICES=0 python code/train_deploy_methods.py --recipe metric35
CUDA_VISIBLE_DEVICES=0 python code/train_deploy_methods.py --recipe predbag2_35
```

Outputs are `experiments/final_candidate_20260904/runs/seed{42,77,2026}/selected.pt`
and `experiments/deploy_methods_20260905/runs/{metric35,predbag2_35}/selected.pt`.
The historic RNG/head construction, selection steps, labels and schedules remain
unchanged. See [the frozen recipe](../review_submission_20260910/TRAINING.md).

## 4. Explicitly launch the SSL + raw sixth fit

Use the absolute path to this repository's adapter:

```bash
CUDA_VISIBLE_DEVICES=0 python /absolute/path/to/clear-ec-research/training_reproduction/train_ssl_raw.py \
  --workspace /absolute/path/to/new-training-workspace \
  --output /absolute/path/to/new-sixth-fit
```

This runs 6,000 SSL updates using all 9,000 images, then 5,215 raw-label regression
updates on 8,998 eligible rows. The terminal teacher encoder initializes fresh
heads. The adapter preserves the original functions, optimizer, seeds, crop rules,
batch sizes and frozen schedule. Output: `pretrain/encoder_step6000.pt` and
`raw_deploy/selected.pt`, with configuration, source hashes, sampling coverage and
strict reload results. `--smoke` runs four SSL and two regression steps using the
original batches; it is still a real CPSAM training job requiring adequate VRAM.

Rerun checkpoints are NOT automatically renamed, calibrated, packaged, promoted or
uploaded. Released inference uses the existing six-member model archive. New fit
bytes and accuracy may differ; neither bitwise refit equality nor a new score is
guaranteed. Recreating the entire historical model-selection search is separate.

## Verification boundary

- Original first-five source checks and latest release source inventory passed.
- Imports work from a directory with no historical `experiments/` tree.
- New workspace initialization passed on the separate server with 9,000 original
  image paths, the original CSV (8,954 exact matches), and verified foundation.
  This does not mean full image caches were rebuilt.
- Safety/input tests and original SSL unit tests passed on the separate Linux venv.
- `smoke_loops.py` passed one SSL and one regression update plus strict reload on
  a tiny synthetic encoder, using original loop batch sizes. This is not CPSAM
  training, a data-pipeline test, or evidence of accuracy.
- Full cache preparation and complete six-member retraining have not been rerun.
  This adapter removes operational dependencies; it does not retroactively certify
  a full training reproduction that was not executed.
