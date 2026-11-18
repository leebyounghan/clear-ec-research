# CLEAR-EC Phase II submission: team Lamda&Kim

Submission of team Lamda&Kim to the CLEAR-EC Grand Challenge, Phase II (final model: tier p3b). Given one corneal endothelium specular microscopy image (972 x 1296,
8-bit `.mha`), the method predicts:

| Output file | Quantity |
|---|---|
| `cell-density.json` | endothelial cell density (CD), cells/mm² |
| `coefficient-of-variation.json` | coefficient of variation of cell area (CV) |
| `hexagonality.json` | fraction of hexagonal cells (HEX) |

The submitted algorithm is the container image built from this repository (`Dockerfile`), run with the checkpoints hosted on Hugging Face
([Byounghan/clear-ec-p3b](https://huggingface.co/Byounghan/clear-ec-p3b); see `checkpoints/README.md`). The method is described in `Technical_Report.pdf` (source: `docs/TECHNICAL_REPORT.md`).

## Repository structure

```
.
├── README.md                  this guide
├── Technical_Report.pdf       method, code organisation, implementation details
├── Dockerfile                 the submitted inference image
├── requirements.txt           pinned Python packages installed into the image
├── constraints.txt            pins for transitive packages
├── src/experiments/           inference code (paths are kept as in the research tree; the image copies src/ to /workspace/clear-ec-research)
├── configs/
│   ├── model_files.txt        the 438 checkpoint paths the code reads
│   ├── checkpoint_sha256.txt  sha256 and size of every checkpoint file (443 files)
│   └── MANIFEST.json          provenance of the build
├── scripts/                   build, run, verify and score helpers
├── checkpoints/               download and placement instructions; the extracted weights go here
├── examples/                  inputs.json template and reference outputs of the three platform debug cases
└── docs/TECHNICAL_REPORT.md
```

Entry point: `src/experiments/lane_cellroute_20261109/inference/docker/gc_entry.py`. Pipeline: `inference/recipe_infer.py` and
`inference/members.py`. Fitted parameters: `inference/manifests/deploy_p3b.json` and `inference/params/deploy_p3b.json`.

## 1. Environment and dependencies

Everything runs inside the Docker image; no Python environment is needed on the host.

- Host: Linux x86-64, Docker 24 or newer, NVIDIA Container Toolkit, an NVIDIA driver that supports CUDA 12.6.
- Build needs internet access for the base image and the pip packages. Running does not (`--network none`).

## 2. Software versions

| Component | Version |
|---|---|
| Base image | `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime`, pinned by digest in `Dockerfile` |
| Python | 3.12 (from the base image) |
| PyTorch / torchvision | 2.11.0+cu126 / 0.26.0+cu126 |
| CUDA / cuDNN | 12.6 / 9 |
| Key packages | timm 1.0.20, transformers 5.3.0, cellpose 4.2.1.1, segment_anything 1.0, SimpleITK 2.5.5, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.9.0, opencv-python-headless 4.11.0.86 |

The full list is `requirements.txt` with `constraints.txt`; the build fails if `pip check` or the version asserts fail. A vendored copy of
timm and the DINOv3 backbone code is in `src/experiments/phase2_foundation_20261002/runtime` and is placed first on `PYTHONPATH`.

## 3. Checkpoints

The 443 checkpoint files (16.25 GB) are public on Hugging Face at [Byounghan/clear-ec-p3b](https://huggingface.co/Byounghan/clear-ec-p3b). Download them into `checkpoints/`
and verify (details in [`checkpoints/README.md`](checkpoints/README.md)):

```bash
pip install -U huggingface_hub
hf download Byounghan/clear-ec-p3b --local-dir checkpoints/ --include "cpsam_models/*" --include "experiments/*"
python3 scripts/verify_checkpoints.py
```

## 4. Input data

The container follows the Grand Challenge interface. For one image it expects:

```
/input/inputs.json                                               [{"socket": {"slug": "corneal-specular-microscopy-image"}}]
/input/images/corneal-specular-microscopy-image/<any name>.mha   972 x 1296, uint8, single channel
```

`scripts/run_case.sh` creates this structure from a plain `.mha` file. No other preparation is needed: the file name is not used, and
non-uint8 or multi-channel inputs are converted to 8-bit grey. Preprocessing (CLAHE, percentile scaling, crops) happens inside the pipeline.

## 5. Commands

```bash
git clone <this repository> && cd <repository>
# place and verify the checkpoints (step 3), then:
scripts/build_image.sh                                   # builds clear-ec-p3b:local, about 5-15 min
scripts/run_case.sh path/to/image.mha outputs/image1      # one image
scripts/run_batch.sh path/to/mha_folder outputs/          # every .mha in a folder, one fresh container per image
```

`run_case.sh` runs the container exactly as the platform does:

```bash
docker run --rm --gpus all --network none --memory 16g --shm-size 1g \
  -v $PWD/checkpoints:/opt/ml/model:ro -v <input_dir>:/input:ro -v <output_dir>:/output clear-ec-p3b:local
```

On an NVIDIA T4 the container selects fp16 autocast and one detector copy by itself. To reproduce that code path on another GPU, add
`--t4path` to `run_case.sh` / `run_batch.sh` (it sets `CLEAR_EC_AUTOCAST=fp16` and `CLEAR_EC_DET_COPIES=1`).

## 6. Outputs

Per image, in the output folder:

| File | Content |
|---|---|
| `cell-density.json`, `coefficient-of-variation.json`, `hexagonality.json` | one JSON float each |
| `run_log.json` | one JSON line: seconds, peak memory, `"mode"` (`recipe` = normal prediction) and `"errors"` |

`run_batch.sh` also writes `outputs/predictions.csv`. If an image cannot be read, has another shape than 972 x 1296, raises an error, or
yields a non-finite or non-positive value, the container writes the constants CD 2770.0, CV 0.36, HEX 0.53 and logs `"mode": "constant"`.

## 7. Hardware and runtime

| | Requirement or measurement |
|---|---|
| GPU | one NVIDIA GPU with at least 8 GB memory; peak use 4.3 GB. Container tested on an RTX 4090; the preceding build with the same code path ran on a Colab Tesla T4. The platform runs a T4. |
| RAM | 16 GB container limit used in tests; peak 5.7 GB |
| Disk | 16.25 GB checkpoints, about 8 GB image |
| Time per image | 68-79 s on an RTX 4090 (T4 code path); estimated 258-296 s on a T4 (each image reads 15.4 GB of weights) |
| 1,000 images | about 20-22 h on an RTX 4090 with one container per image |

## 8. Reproducing the submitted results

1. Build the image and place the checkpoints (steps 3 and 5).
2. Check the three platform debug cases against `examples/expected_outputs_debug_cases.csv`. With `--t4path` the values should agree
   within 0.1 % for CD and 0.5 % for CV and HEX; GPU kernels are not bit-deterministic.

   | Case | CD | CV | HEX |
   |---|---:|---:|---:|
   | 0001-19 | 2854.7 | 0.3819 | 0.5462 |
   | 0001-20 | 1999.2 | 0.4796 | 0.4134 |
   | 0001-21 | 4373.7 | 0.2977 | 0.6496 |

3. Run `scripts/run_batch.sh <test_images> outputs/ --t4path` on the hidden test images. If labels are available as a CSV with columns
   `case, cell_density, coefficient_of_variation, hexagonality`, score with the organisers' rule:

   ```bash
   python3 scripts/score.py outputs/predictions.csv labels.csv
   ```

| MAPE (%) | Mean | CD | CV | HEX |
|---|---:|---:|---:|---:|
| Out-of-fold, 8,954 training images (five donor-grouped folds) | 8.6596 | 6.1479 | 10.1115 | 9.7194 |
| Phase II hidden test set (1,000 images) | 8.9352 | 6.1624 | 10.2901 | 10.3532 |

Details are in `Technical_Report.pdf`.

## Artifacts of the submitted version

| File | Bytes | sha256 |
|---|---:|---|
| Container image `clear-ec_20261114-freeze-9687076-p3b.tar.gz` (`docker save`, image id `db390738df95`) | 4,114,587,222 | `67a8452f62fbced75c6a3499e2c9b068ac0802e306b177381f73c154d3cd96c1` |
| Checkpoint archive `model_freeze-9687076-p3b_20261114.tar.gz` (submitted; its 443 files are the ones on Hugging Face) | 15,056,963,340 | `a51a21581409b9cd7c527d90e8f4df6af67f268f575278a3583c15dfbbe3bf1d` |

An image built from this repository differs in its id from the submitted one (build time stamps, pip mirrors) but contains the same code.
