> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../../docker_review_20260914/README.md) for the release-facing summary.

# Lamda&KIM — Docker inference, September 14 model

This is the primary reviewer execution path. **No Kubernetes, kubectl, kubeconfig,
private namespace, PVC, training container or host Python/venv is required.**
All model loading and inference run inside the archived Docker image. The host
only downloads/verifies/extracts artifacts and mounts input, output and models.

This uses the delivered sixth-only standard SSL + raw-label release, NOT the
later all-five SSL retraining. The team reported Phase I score8.7010; exact platform
model-ID/archive identity is still a separate review check. See
[version map](../docs/STATUS_20260915.md). Model hosting URLs are still pending;
this guide is not yet an access-complete technical submission.

## 1. Host requirements

- Linux x86_64 with Docker Engine and NVIDIA Container Toolkit GPU support.
- NVIDIA GPU and driver compatible with CUDA12.6. The challenge allocation was
  one T4/16GiB VRAM and16GB host RAM; actual T4 runtime remains unmeasured here.
- Bash, GNU coreutils (`sha256sum`, `realpath`) and `tar`.
- Disk for ~12GB archives, ~8.55GB extracted models and the loaded Docker image.

The image contains Python3.12.3, PyTorch2.11.0+cu126 and application dependencies.
Do not install training packages on the host. macOS CPU/MPS is not a replacement
for this CUDA path. See [Docker GPU access](https://docs.docker.com/engine/containers/resource_constraints/#gpu).
Historical H200 checks measured ~8–10seconds/image; this is not a T4 guarantee.

## 2. Download and verify exact artifacts

Run from this directory (`cd docker_review_20260914` after cloning the research
repository). Download these two files into a new `artifacts/` folder from the
reviewer-accessible hosting links once published:

- `clear_ec_full_quality77_20260909.tar.gz`: unchanged container archive.
- `model_full_quality77_ssl_raw_20260914.tar.gz`: latest delivered model archive.

Do not substitute `model_full_quality77_20260909.tar.gz`, the raw-only September13
archive, or newly retrained first-five weights. Check before extraction/loading:

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

`mkdir checkpoints/model` intentionally refuses an existing directory. Do not
unpack over another release. Docker image ID and archive SHA256 are different.
Only use the checksum-verified trusted archives. The original image and all
inference sources remain unchanged; models are mounted separately.

## 3. Prepare one input and a new output directory

```bash
mkdir -p test_input/images/corneal-specular-microscopy-image
cp inputs.json test_input/inputs.json
cp /absolute/path/to/authorized-case.mha \
  test_input/images/corneal-specular-microscopy-image/case.mha
mkdir test_output
sudo chown 10001:10001 test_output
```

Use exactly one self-contained grayscale uint8 MHA, with at least512x512 pixels.
Do not resize/rescale cell sizes or supply manual ROI. Input files and model
files must be readable and their directories traversable by container UID10001;
grant access only to this dedicated review workspace. Output must be empty and
writable by UID10001. Do not run broad recursive permission changes.

## 4. Execute inside Docker

```bash
bash run.sh test_input checkpoints/model test_output 0
```

The fourth argument is the host GPU index (default0). The runner verifies all eight
model-resource hashes and the loaded image ID, then runs the original entrypoint
with FP16. It mounts only input/model read-only and output writable, uses a
read-only root filesystem, temporary `/tmp`, and `--network none`. It does not
mount the repository, Docker socket, home directory or any cluster configuration.

Outputs:

- `test_output/cell-density.json`: one finite positive number, cells/mm².
- `test_output/coefficient-of-variation.json`: one finite positive fraction.
- `test_output/hexagonality.json`: one finite positive fraction.

Do not multiply CV or HEX by100. Use a new empty output directory for each case.
The three output filenames and calculation are the unchanged challenge contract.

## 5. Source rebuild (optional, different image identity)

From the repository root:

```bash
docker build --platform linux/amd64 -t lamda:review-rebuild review_submission_20260910
```

That Dockerfile uses the public pinned PyTorch base and copies only inference
`app/` and dependency constraints. Its `.dockerignore` excludes research/K8s files,
models and documentation from build context. Build-time internet access may be
needed; inference with the complete model archive needs no network. A rebuild is
not guaranteed to have the archived image ID. The exact-image runner above
intentionally does not accept an unverified rebuild as the submitted container.

## 6. Verification scope and repository separation

This update changes the reviewer workflow, not model predictions. Script syntax,
resource-manifest consistency and command construction can be checked without
training. **No new end-to-end Docker GPU execution is claimed by this update.**
A container smoke test on a Linux NVIDIA host remains desirable; reproducing our
private Kubernetes infrastructure or performing a new K8s venv test is not required.
External model access, updated technical PDF and any full-training reproduction
gaps remain separate checklist items.

The [revised technical PDF](../../../../output/pdf/Lamda_KIM_CLEAR_EC_Technical_Report.pdf)
now describes this release, its evidence limits, and a dated experiment appendix.
Its editable builder is `scripts/build_technical_report.py` (Python + reportlab;
document generation only, no GPU or model loading). The September 10 PDF remains
unchanged. Updating the PDF does not close platform identity/access or runtime gates.

The parent research repository intentionally preserves `k8s/` job manifests with
our historical namespace/PVC/node-selector settings. They are not kubeconfig
credentials and are not a dependency of this runner or final inference image.
Historical `review_submission_20260910` venv instructions/provenance are optional
records, not steps to execute for this Docker workflow.
