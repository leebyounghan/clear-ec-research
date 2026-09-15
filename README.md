# Lamda&KIM — CLEAR-EC Phase I

Code and documentation for the September 14, 2026 submission. The model predicts
cell density (CD), coefficient of variation (CV) and hexagonality (HEX) from a
single corneal specular microscopy image.

| Resource | Link |
|---|---|
| Technical report, including experiment timeline | [PDF](output/pdf/Lamda_KIM_CLEAR_EC_Technical_Report.pdf) |
| Model weights and Docker image | [Google Drive](https://drive.google.com/drive/folders/1QyNtGhaEpePwBEyjj-UorqRkSFDRmKUn?usp=sharing) |
| Installation and inference | [Docker guide](docker_review_20260914/README.md) |
| Optional verified source execution | [Fresh venv guide and result](docs/VENV_SMOKE_20260915.md) |
| Optional standalone training | [Training workspace and commands](training_reproduction/README.md) |
| Model recipe and training evidence | [Method](docs/METHODOLOGY.md) · [Release record](docs/SSL_RELEASE_20260914.md) |
| File identities | [Artifact manifest](docs/ARTIFACTS.md) |

## Results

| Submission | Phase I score |
|---|---:|
| September 14 | [8.7010](https://clear-ec.grand-challenge.org/evaluation/7606a2b6-818f-46d0-a1fc-d060648fc2c6/) |
| September 10 baseline | [8.7042](https://clear-ec.grand-challenge.org/evaluation/0a82fe58-7f9a-4a88-b744-bcfa84b1e43a/) |

Lower is better. Official evaluation 8.7010, its model/image versions, and the shared
archive checksums have been verified against the platform records. Internal scores
are reported separately. See the [verified release binding](docs/ARTIFACTS.md#official-evaluation-and-shared-files).

## Method

Six equally weighted Cellpose-SAM regression models use native 256 × 256 tiles
and 40 views per member. Five sample the central field; the sixth uses
quality-selected regions across the original image. Predictions are decoded
before ensemble averaging and fixed per-metric calibration.

The September 14 package retains the first five checkpoints and replaces only
the sixth with custom masked-EMA SSL followed by raw-label regression:
9,000 training images / 6,000 SSL updates, then 8,998 valid labeled images /
5,215 regression updates. The container and calibration are unchanged.

## Run inference

Use Linux x86_64 with Docker, NVIDIA Container Toolkit and a CUDA 12.6-compatible
GPU driver. No Kubernetes or host Python environment is required.

1. Clone this repository.
2. Follow the [Docker guide](docker_review_20260914/README.md) to download and
   verify the archives, load the image, and prepare one MHA input.
3. Run:

```bash
cd docker_review_20260914
bash run.sh test_input checkpoints/model test_output 0
```

The container writes three JSON numbers: CD in cells/mm², CV and HEX as fractions.
Models and input are mounted read-only; output must be a new, writable directory.

## Repository map

- `docker_review_20260914/`: current Docker instructions, runner and checksums.
- `review_submission_20260910/`: unchanged inference source and historical training recipes.
- `output/pdf/`: current technical report.
- `docs/`: concise method, artifact and development summaries.
- `docs/archive/`: historical research/operations records, not reviewer setup instructions.
- `code/`, `results/`, `k8s/`: research code, aggregate evidence and historical job manifests.

Challenge images and large model binaries are not stored in Git. See
[reproduction scope](docs/REPRODUCE.md) for validation limits and the
[review checklist](docs/NEXT_STEPS.md) for outstanding access checks.
