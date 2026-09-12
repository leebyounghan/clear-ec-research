# Lamda - CLEAR-EC Phase I technical review

Local review draft for the package prepared on 9 September UTC / 10 September KST, 2026.
Algorithm name: **Lamda**. Team name: **pending confirmation**; Lamda is the proposed matching team name.

This directory preserves the submitted candidate, not the latest research branch.
It is not yet an access-complete submission: see `RELEASE_CHECKLIST.md` before sharing.
The official [September 10 evaluation](https://clear-ec.grand-challenge.org/evaluation/0a82fe58-7f9a-4a88-b744-bcfa84b1e43a/)
reports **Succeeded**, score **8.7042**, and leaderboard position **1** when checked
on September 12, 2026 (rank can change). User: Leebyounghan; algorithm: Lamda.
Image version: `730ff3f7-ea96-4180-936d-bc02a4eca762`.
Model version: `db061942-478a-42b6-a6d7-b6dff1626108`.
These are platform version IDs, not archive checksums. The page does not expose
per-metric scores or independently establish byte identity with the local archives.

[Read the technical report (PDF)](output/pdf/Technical_Report.pdf).
The existing PDF is preserved; the newly verified official result is recorded here.

## Quick start: Python venv, no Docker required

**Supported host:** Linux x86_64, Python 3.12 with `venv`/pip, and an NVIDIA GPU
with a working CUDA 12.6-compatible driver (`nvidia-smi` must work). Install OS
Python/venv prerequisites using your host's package manager if missing. This is
an **inference-only** environment, not the historical CUDA 13 training environment.
The local macOS CPU/MPS path is unsupported. A separate CUDA toolkit/compiler is
not required for the prebuilt PyTorch wheels; the NVIDIA driver is required.

Start from this directory. In the research checkout, first run
`cd review_submission_20260910`; in a standalone review repository, use its root.

### A. Install the isolated environment

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
python scripts/check_environment.py
```

The setup script installs PyTorch 2.11.0+cu126 and torchvision 0.26.0+cu126 from
the official PyTorch index, then the pinned application dependencies under the
exported runtime constraints. It never uses `sudo pip`, system-site packages or
`--break-system-packages`. If Python 3.12 has another executable path, use
`LAMDA_PYTHON=/path/to/python3.12 bash scripts/setup_venv.sh`.

Equivalent manual installation, in a fresh environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0+cu126 torchvision==0.26.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-venv.txt
python -m pip check
python scripts/check_environment.py
```

The official wheel version mapping is documented by
[PyTorch](https://pytorch.org/get-started/previous-versions/#v2110).

### B. Obtain and verify the model (Docker image download is optional)

Download `model_full_quality77_20260909.tar.gz` using
[the checkpoint distribution instructions](checkpoints/README.md) and place it
under `artifacts/`. **Hosting URLs are not published yet**; this is the remaining
download prerequisite, not a working download link. Do not use a later model.

```bash
python scripts/verify_release.py \
  --model-archive artifacts/model_full_quality77_20260909.tar.gz
mkdir checkpoints/model
tar -xzf artifacts/model_full_quality77_20260909.tar.gz -C checkpoints/model
python scripts/verify_release.py --model checkpoints/model
```

`mkdir` intentionally fails if the model directory already exists: do not unpack
over an existing release. If the models are already present, skip extraction
and run the final verification command only. The 7.96 GB archive expands to
about 8.55 GB; allow additional disk space for the venv and installation cache.
You do **not** need the 4.06 GB Docker image archive for this venv path.

### C. Predict from one original MHA image

Replace only the image path below with an authorized original image. The output
directory must be new or empty; your image is copied byte-for-byte into a temporary
Grand Challenge input layout and removed from that temporary location afterward.

```bash
python scripts/predict_image.py \
  --image /absolute/path/to/case.mha \
  --model checkpoints/model \
  --output predictions/case001 \
  --gpu 0
```

The wrapper checks environment and frozen code identity, then runs the unchanged
submission entrypoint with the exact configuration and `--fp16`. Full model SHA256
verification is done in step B, not repeated for every image. No input resizing,
new calibration or model conversion is introduced. It writes:

```text
predictions/case001/cell-density.json
predictions/case001/coefficient-of-variation.json
predictions/case001/hexagonality.json
```

Each file contains one positive finite number. CD is cells/mm²; CV and HEX are
fractions, not percentages. Use a different output folder for each image.
`--gpu` selects a device explicitly; use indices visible to the process/container.

Already have the Grand Challenge input layout? Run the original entrypoint directly:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python app/inference.py --input test_input --output predictions/case002 \
  --model checkpoints/model --config candidate6_full_quality.json --fp16
```

Environment import checks are not numerical parity certification. A venv preserves
the model and inference code but not the entire submitted OS image. Full clean-venv
GPU comparison against the archived image is still pending; use the archived image
below when exact container identity is required. Offline flags disable supported
Hub downloads; they are not an operating-system network isolation policy.

**Installation verified in Kubernetes:** a fresh Linux x86_64 Python 3.12.14 venv
passed installation, `pip check`, imports, all eight synthetic wrapper tests and
hash checks for the eight model resources on 12 September 2026. The successful
verification container took 135 seconds. No GPU was requested; this is not a GPU
inference benchmark. See `provenance/k8s_venv_verification.json` for the image digest
and exact checked versions. Original Docker Python is 3.12.3, so do not claim a
byte-identical full environment from the venv test.

### Common setup issues

- `No module named venv` / `ensurepip`: install your OS's Python 3.12 venv support.
- CUDA unavailable: check `nvidia-smi`, driver compatibility, GPU allocation and
  `CUDA_VISIBLE_DEVICES`. Do not switch to CPU/BF16 to bypass a failed CUDA check.
- Wrong package version: use a separate clean Python 3.12 venv rather than merging
  this recipe into a training or system environment.
- Missing `cpsam_v2`: extract the complete model archive, not just the six fitted files.
- Existing output directory with files: select a new directory; results are never overwritten.
- Out of memory: ensure the chosen GPU is not occupied; do not silently change the
  frozen view batch or preprocessing and label it the same validated submission.

## 1. What is included

- `app/`, `Dockerfile`, `constraints.txt`: byte-preserved September package inference sources.
- `configs/candidate6_full_quality.json`: byte-preserved six-member configuration.
- `provenance/`: archive identity, dependency lock and historical training configurations.
- `training/`: historical all-data fitting and preparation sources, kept separately from inference.
- `scripts/verify_release.py`: source/configuration/checkpoint identity checks.
- `scripts/setup_venv.sh`, `check_environment.py`, `predict_image.py`: venv setup,
  import/CUDA checks and single-image execution without changing the frozen app.
- `TRAINING.md`: historical data preparation and fitting sequence, including remaining validation gates.
- `output/pdf/Technical_Report.pdf`: technical description and evidence limitations.

No challenge images, labels, credentials or large checkpoint binaries are included in this code directory.
The original model archive contains the foundation checkpoint, six fitted checkpoints and configuration.
New JEPA/SSL, artifact-guard and raw-target experiments are **not** part of this version.

## 2. Exact artifacts and checkpoint placement

Download instructions and archive identities are in `checkpoints/README.md`.
The external download URLs must be completed and reviewer access tested before release.
Once both verified archives have been obtained, place them in `artifacts/` under this directory.

```bash
python3 scripts/verify_release.py --archives artifacts
mkdir -p checkpoints/model
tar -xzf artifacts/model_full_quality77_20260909.tar.gz -C checkpoints/model
python3 scripts/verify_release.py --model checkpoints/model
docker load -i artifacts/clear_ec_full_quality77_20260909.tar.gz
docker image inspect clear-ec:full-quality77-20260909 --format '{{.Id}}'
```

Expected image ID:
`sha256:275803d816ef934ef6665a12c693a2cd31bc1e418fdb734a7672a51404111a7d`.
The image archive checksum and Docker image ID are different identifiers.
Only extract the checksum-verified archive from the trusted release source.

## 3. Software and hardware

Use the original exported Linux AMD64 image to reproduce inference. It contains
Python 3.12.3, PyTorch 2.11.0+cu126, CUDA 12.6 and cuDNN 9. Direct dependencies are
pinned in `app/requirements.txt`; the exported environment is recorded in
`provenance/runtime-lock.txt`. The Dockerfile pins its base image by digest.

Inference requires Linux, an NVIDIA driver compatible with CUDA 12.6, and a CUDA GPU.
Docker with NVIDIA GPU support is needed only for the container path. The requested challenge allocation was one T4 with
16 GiB GPU memory and 16 GB host memory. Actual T4 capacity and runtime have not
been independently verified in this local review. Apple laptop CPU execution is
not a substitute for the CUDA entrypoint.

Historical H200 checks with the same pinned base and inference dependencies measured
about 8.1 seconds/image in a warm process including member loading, 10.2 seconds
for one separate-process image, and 2.64 GiB peak allocated tensors. These are
H200 observations, not T4 estimates or total process/device memory requirements.
Allow space for both compressed archives (~12 GB), extracted models (~8.55 GB),
the loaded Docker image and normal Docker storage overhead.

## 4. Input layout

Run one image at a time, using an empty output directory:

```text
test_input/
  inputs.json
  images/corneal-specular-microscopy-image/case.mha
test_output/                         # empty, writable by UID 10001
checkpoints/model/
  cpsam_v2
  seed42.pt
  seed77.pt
  seed2026.pt
  metric35.pt
  predbag2_35.pt
  full_quality_seed77.pt
  candidate6_full_quality.json
```

`inputs.json` contains:

```json
[{"socket": {"slug": "corneal-specular-microscopy-image"}}]
```

Use an authorized, self-contained, grayscale uint8 MHA. The historical training
images were 972 x 1296 pixels. Do not resize the image or rescale the cells before
inference. The reader also accepts a singleton leading image dimension; the
central field must be at least 512 x 512 pixels. No manual ROI is supplied.

## 5. Execute the original image

Run from this directory on the Linux GPU host. These commands intentionally set
permissions only on a new, empty test output directory.

```bash
mkdir test_output
sudo chown 10001:10001 test_output
docker run --rm --gpus '"device=0"' --network none --read-only \
  --platform linux/amd64 --user 10001:10001 \
  --cpus 4 --memory 16g --shm-size 1g \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --mount "type=bind,source=$PWD/test_input,target=/input,readonly" \
  --mount "type=bind,source=$PWD/test_output,target=/output" \
  --mount "type=bind,source=$PWD/checkpoints/model,target=/opt/ml/model,readonly" \
  clear-ec:full-quality77-20260909
```

The fixed entrypoint runs `inference.py --config candidate6_full_quality.json --fp16`.
It rejects multiple inputs, unexpected member order/configuration, incompatible
checkpoint structures and existing output files. It loads the six members
sequentially with eight views per forward batch, without downloading weights.

Expected output files, each containing one finite positive JSON number:

- `test_output/cell-density.json`: cell density (cells/mm²).
- `test_output/coefficient-of-variation.json`: CV in the dataset's fractional units.
- `test_output/hexagonality.json`: HEX in the dataset's fractional units.

Do not multiply CV or HEX by 100. Use a new empty output directory for each case.
Evaluation of these outputs against private challenge labels remains the organizer's job.

## 6. Source rebuild versus exact archived image

```bash
docker build --platform linux/amd64 -t lamda:review-rebuild .
```

This builds the included inference sources. A rebuild need not have the same image
digest or bitwise outputs: transitive dependencies and the host runtime must also
be checked against the archived lock. For exact submitted-image identity, use
the original exported archive, not a newly built image.

## 7. Validation status

The new venv installation path passed the Kubernetes CPU validation described
above. A first source-bundle check encountered macOS AppleDouble sidecars; the
transfer excludes them and training configuration verification now uses explicit
file membership. The clean retry passed without altering the submitted app/model.

Historical evidence: exported-image offline CPU import/preprocessing/checkpoint
smokes passed; pinned-runtime H200 FP16 outputs matched eight historical reference
cases. The H200 test did **not** use the actual exported image under enforced
network isolation. Full offline CUDA execution of this review checkout on T4 is
still a release gate. No clean full retraining has been performed for this review.

`provenance/local_verification.json` records the local checks performed during
organization. Passing hashes does not certify numerical reproduction on a new GPU.
Historical cross-validation numbers and post-submission experiments must not be
presented as the private leaderboard result.
