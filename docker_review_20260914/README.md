# Docker inference — September 14 release

Run the original container with the separately mounted SSL + raw-label model.
No Kubernetes or host Python/venv setup is required.

## Requirements

- Linux x86_64, Docker Engine and NVIDIA Container Toolkit.
- NVIDIA GPU with a CUDA 12.6-compatible driver.
- Bash, GNU `realpath` / `sha256sum`, and `tar`.
- Disk for approximately 12 GB of archives, 8.55 GB of extracted models,
  the loaded Docker image and Docker storage overhead.

The image contains Python 3.12.3 and PyTorch 2.11.0+cu126. The challenge allocation
is a T4 with 16 GiB VRAM and 16 GB RAM; independent T4 timing is not recorded.

## 1. Download and verify

```bash
git clone https://github.com/leebyounghan/clear-ec-research.git
cd clear-ec-research/docker_review_20260914
mkdir artifacts
```

Private-repository access must already be granted. From the
[Google Drive folder](https://drive.google.com/drive/folders/1QyNtGhaEpePwBEyjj-UorqRkSFDRmKUn?usp=sharing),
save these exact files into `artifacts/`:

- `clear_ec_full_quality77_20260909.tar.gz`
- `model_full_quality77_ssl_raw_20260914.tar.gz`

The older date in the Docker filename is expected: only the model archive changed.

```bash
set -euo pipefail
(cd artifacts && sha256sum --check ../SHA256SUMS.txt)
mkdir -p checkpoints
mkdir checkpoints/model
tar --no-same-owner -xzf artifacts/model_full_quality77_ssl_raw_20260914.tar.gz \
  -C checkpoints/model
(cd checkpoints/model && sha256sum --check ../../MODEL_SHA256SUMS.txt)
docker load -i artifacts/clear_ec_full_quality77_20260909.tar.gz
docker image inspect \
  sha256:275803d816ef934ef6665a12c693a2cd31bc1e418fdb734a7672a51404111a7d \
  --format '{{.Id}}'
```

Do not extract over an existing model directory. Archive SHA256 and Docker image
ID are different identifiers; see [artifact identities](../docs/legacy_phase1/ARTIFACTS.md).

## 2. Prepare one input

Use one authorized, self-contained grayscale uint8 MHA, at least 512 × 512 pixels.
Preserve its native resolution; do not resize cells or supply manual ROI.

```bash
mkdir -p test_input/images/corneal-specular-microscopy-image
cp inputs.json test_input/inputs.json
cp /absolute/path/to/authorized-case.mha \
  test_input/images/corneal-specular-microscopy-image/case.mha
mkdir test_output
sudo chown 10001:10001 test_output
```

Replace the example image path. Input/model files must be readable and parent
directories traversable by container UID 10001. Output must be empty and writable
by that UID; restrict permission changes to this dedicated directory.

## 3. Execute

```bash
bash run.sh test_input checkpoints/model test_output 0
```

The final argument selects the GPU (default 0). The runner checks all eight model
resource hashes and the image ID, then runs FP16 inference with networking disabled.
It mounts only input/model read-only and output writable.

| Output in `test_output/` | Value |
|---|---|
| `cell-density.json` | Positive finite number, cells/mm² |
| `coefficient-of-variation.json` | Positive finite fraction |
| `hexagonality.json` | Positive finite fraction |

Do not multiply CV or HEX by 100. Use a new empty output directory for every case.

## Source and verification

The unchanged source and pinned dependencies are in
[`review_submission_20260910/`](../review_submission_20260910/README.md).
An optional source rebuild from the repository root is:

```bash
docker build --platform linux/amd64 -t lamda:review-rebuild review_submission_20260910
```

A rebuild has a different identity and is not accepted by the exact-image runner
above. Use the verified archive for release reproduction.

Hashes, strict reload and eight-image frozen CUDA 12.6 application checks passed.
On September 15, a fresh GitHub checkout and complete anonymous Drive downloads
passed archive/resource hashes and one-image archived-container inference on an
RTX 4090; see the [verification record](../docs/legacy_phase1/DOCKER_SMOKE_20260915.md).
This does not certify T4 execution, accuracy, or evaluator GitHub access.
See [reproduction scope](../docs/legacy_phase1/REPRODUCE.md)
and the [technical report](../output/pdf/Lamda_KIM_CLEAR_EC_Technical_Report.pdf).
