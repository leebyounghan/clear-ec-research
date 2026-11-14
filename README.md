# CLEAR-EC Phase II: tier p3b

Final model of this project for the CLEAR-EC Grand Challenge, Phase II. Given one corneal endothelium specular microscopy image
(972 x 1296, 8-bit), it predicts three values:

| Output | Meaning |
|---|---|
| CD | endothelial cell density, cells/mm² |
| CV | coefficient of variation of cell area |
| HEX | hexagonality, the fraction of six-sided cells |

The challenge score is the mean of the three per-metric MAPE values. This repository holds the inference code and the build context of
the p3b container image. Model weights and data are not in the repository.

## Results

Out-of-fold MAPE in percent, five folds grouped by donor. Every fitted quantity (blend weights, scalars, stretch, model choices) was
fitted without the scored fold.

| Model | Mean | CD | CV | HEX |
|---|---:|---:|---:|---:|
| **p3b, out-of-fold** | **8.6596** | 6.1479 | 10.1115 | 9.7194 |
| Tier p (previous build), out-of-fold | 8.6682 | 6.1737 | 10.1115 | 9.7194 |
| Submission 1 (earlier model), out-of-fold | 8.877 | 6.312 | 10.204 | 10.115 |
| Submission 1, hidden test set (1,000 images) | 8.9675 | 6.250 | 10.143 | 10.510 |

- p3b against tier p on CD: -0.0258 (95 % CI -0.040 to -0.012), lower in every fold. On a repeated grouped 10-fold fit with a
  donor-clustered bootstrap the difference is -0.0306 (CI -0.045 to -0.016).
- CV and HEX are the same as in tier p.
- No test-set result for p3b is recorded here.

## Repository layout

| Path | Content |
|---|---|
| `p3b/Dockerfile` | image definition; build with `--build-arg TIER=p3b` |
| `p3b/code/` | inference code at the paths the image expects |
| `p3b/code/experiments/lane_cellroute_20261109/inference/` | entry point (`docker/gc_entry.py`), pipeline (`recipe_infer.py`, `members.py`), manifests and fitted parameters |
| `p3b/model_files.txt` | weight paths the image links into the model mount |
| `p3b/MANIFEST.json` | provenance of the build context |
| `p3b/requirements.txt`, `p3b/constraints.txt` | pinned Python packages |
| `docs/METHOD.md` | data, model components, evaluation protocol |

## Build and run

```bash
docker build --build-arg TIER=p3b -t clear-ec:<YYYYMMDD>-freeze-9687076-p3b p3b/
```

```bash
docker run --rm --gpus all --network none --memory 16g \
  -v <case>/input:/input:ro -v <case>/output:/output -v <model_dir>:/opt/ml/model:ro \
  clear-ec:<YYYYMMDD>-freeze-9687076-p3b
```

- Input: one `.mha` file in `/input/images/corneal-specular-microscopy-image/`.
- Output: `/output/cell-density.json`, `/output/coefficient-of-variation.json`, `/output/hexagonality.json`.
- The weights are mounted read-only at `/opt/ml/model`. The container runs as uid 10001 and needs no network.
- One container handles one image.

## Artifacts

| File | Bytes | sha256 |
|---|---:|---|
| `clear-ec_20261114-freeze-9687076-p3b.tar.gz` (image `db390738df95`) | 4,114,587,222 | `67a8452f62fbced75c6a3499e2c9b068ac0802e306b177381f73c154d3cd96c1` |
| `model_freeze-9687076-p3b_20261114.tar.gz` (443 files, 16.25 GB unpacked) | 15,056,963,340 | `a51a21581409b9cd7c527d90e8f4df6af67f268f575278a3583c15dfbbe3bf1d` |

The image and the model archive belong together. An image of another tier with this archive does not find its weights and returns the
fallback constants.

## Measured behaviour

- Container test on an RTX 4090 (no network, 16 GB RAM limit, uid 10001, a fresh container per image): 17 of 17 jobs succeeded.
- Time per image: 68.0-78.6 s on the RTX 4090 with the T4 code path forced (fp16, one detector copy). On a T4 the estimate is
  258-296 s, derived from stage timings of tier p on two Colab T4 machines. The time on the challenge platform was not measured.
- Peak memory: 5.7 GB RAM, 4.3 GB GPU.
- Fallback: an unreadable input, an image that is not 972 x 1296, any exception, or a non-finite or non-positive output makes the
  container write the constants CD 2770.0, CV 0.36, HEX 0.53 for that image.

Reference outputs for the three platform debug cases (tolerance CD 0.1 %, CV and HEX 0.5 %):

| Case | CD | CV | HEX |
|---|---:|---:|---:|
| 0001-19 | 2854.7 | 0.3819 | 0.5462 |
| 0001-20 | 1999.2 | 0.4796 | 0.4134 |
| 0001-21 | 4373.7 | 0.2977 | 0.6496 |
