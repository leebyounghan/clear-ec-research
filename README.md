# Lamda&KIM — CLEAR-EC

## Phase II (current)

Start at the [brief](docs/BRIEF.md). Earlier state documents and results were removed from the working tree on 2026-11-06; tracked versions remain in git history up to commit `f8bc49b`.

| Reference | Leaderboard | OOF |
|---|---:|---:|
| Submission 1, hybrid B, 2026-11-03 | 8.9675 | 8.877 |

Lower is better. Results are compared with our Phase II submissions only. Phase I scores, including the Phase I first place, are not a yardstick: they came from another test set.

## Legacy: Phase I release

Closed. [Legacy index](docs/legacy_phase1/README.md). Everything below describes the Phase I release and stays for its reviewers.

Code and documentation for the September 14, 2026 submission. The model predicts
cell density (CD), coefficient of variation (CV) and hexagonality (HEX) from a
single corneal specular microscopy image.

| Resource | Link |
|---|---|
| Technical report, including experiment timeline | [PDF](output/pdf/Lamda_KIM_CLEAR_EC_Technical_Report.pdf) |
| Model weights and Docker image | [Google Drive](https://drive.google.com/drive/folders/1QyNtGhaEpePwBEyjj-UorqRkSFDRmKUn?usp=sharing) |
| Installation and inference | [Docker guide](docker_review_20260914/README.md) |
| Optional verified source execution | [Fresh venv guide and result](docs/legacy_phase1/VENV_SMOKE_20260915.md) |
| Optional standalone training | [Training workspace and commands](training_reproduction/README.md) |
| Model recipe and training evidence | [Method](docs/legacy_phase1/METHODOLOGY.md) · [Release record](docs/legacy_phase1/SSL_RELEASE_20260914.md) |
| File identities | [Artifact manifest](docs/legacy_phase1/ARTIFACTS.md) |

### Phase I results

| Submission | Phase I score |
|---|---:|
| September 14 | [8.7010](https://clear-ec.grand-challenge.org/evaluation/7606a2b6-818f-46d0-a1fc-d060648fc2c6/) |
| September 10 baseline | [8.7042](https://clear-ec.grand-challenge.org/evaluation/0a82fe58-7f9a-4a88-b744-bcfa84b1e43a/) |

Lower is better. Official evaluation 8.7010, its model/image versions, and the shared
archive checksums have been verified against the platform records. Internal scores
are reported separately. See the [verified release binding](docs/legacy_phase1/ARTIFACTS.md#official-evaluation-and-shared-files).

### Phase I method

Six equally weighted Cellpose-SAM regression models use native 256 × 256 tiles
and 40 views per member. Five sample the central field; the sixth uses
quality-selected regions across the original image. Predictions are decoded
before ensemble averaging and fixed per-metric calibration.

The September 14 package retains the first five checkpoints and replaces only
the sixth with custom masked-EMA SSL followed by raw-label regression:
9,000 training images / 6,000 SSL updates, then 8,998 valid labeled images /
5,215 regression updates. The container and calibration are unchanged.

### Phase I inference

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

### Phase I repository map

- `docker_review_20260914/`: Docker instructions, runner and checksums of the September 14 release.
- `review_submission_20260910/`: unchanged inference source and historical training recipes.
- `output/pdf/`: Phase I technical report.
- `docs/legacy_phase1/`: concise method, artifact and development summaries.
- `code/`, `results/`, `k8s/`: research code, aggregate evidence and historical job manifests.

Challenge images and large model binaries are not stored in Git. See
[reproduction scope](docs/legacy_phase1/REPRODUCE.md) for validation limits and the
[review checklist](docs/legacy_phase1/NEXT_STEPS.md) for outstanding access checks.
