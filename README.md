# CLEAR-EC — Research & Training Repo

개발 기록·제출본·재현 코드·데이터/PVC 위치는 [개발 기록 및 재현 자료 인덱스](docs/DEVLOG_INDEX.md)를 참고하세요.

> **Latest reported Phase I result: 8.7010 (September 14, Lamda&KIM).**
> [Evaluation](https://clear-ec.grand-challenge.org/evaluation/7606a2b6-818f-46d0-a1fc-d060648fc2c6/).
> See [September 15 status and version map](docs/STATUS_20260915.md) and
> [SSL + raw-label release](docs/SSL_RELEASE_20260914.md). The prepared release
> replaces only the sixth member; the separately retrained first five are NOT
> part of that reported submission. Score/rank were supplied by the team, not
> freshly verified against platform model IDs in this repository update.
>
> **Frozen September 10 baseline:** use the
> [review-package README](review_submission_20260910/README.md) for the frozen
> six-member model, **Linux NVIDIA GPU + Python venv quick start**, Docker execution
> and checkpoint instructions. The model archive is separate from the image.
> Official Phase I score: **8.7042**; leaderboard position **1** when checked on
> September 12, 2026. [Evaluation](https://clear-ec.grand-challenge.org/evaluation/0a82fe58-7f9a-4a88-b744-bcfa84b1e43a/).
> [Technical report (PDF)](review_submission_20260910/output/pdf/Technical_Report.pdf).
> The review directory is still a release draft; hosting/access and GPU parity
> validation must be completed before technical-review dispatch.

## September 10 baseline inference quick start (Linux NVIDIA GPU)

Python 3.12 and a CUDA 12.6-compatible NVIDIA driver are required. From this clone:

```bash
cd review_submission_20260910
bash scripts/setup_venv.sh
source .venv/bin/activate
python scripts/check_environment.py
```

Download and verify the unchanged model archive following
[checkpoint placement](review_submission_20260910/README.md#b-obtain-and-verify-the-model-docker-image-download-is-optional).
The external download URL is still awaiting publication. Once all eight resources
are present in `checkpoints/model/`:

```bash
python scripts/predict_image.py --image /absolute/path/to/case.mha \
  --model checkpoints/model --output predictions/case001 --gpu 0
```

This produces three scalar JSON files (CD, CV, HEX) without Docker. The original
submitted inference sources and model bytes are unchanged; clean-venv GPU parity
is not yet certified. For the original archived image, follow the review README's
Docker path. Older research scripts and documents remain for historical reference.

Training, evaluation, and analysis code for the [CLEAR-EC Grand Challenge](https://clear-ec.grand-challenge.org/)
(Corneal Learning for Endothelial Assessment and Review). The task: predict three clinical metrics per
specular-microscopy image — **Cell Density (CD)**, **Coefficient of Variation (CV)**, **Hexagonality (HEX)**.

This repo contains research/training history and the frozen September 10 review
package under `review_submission_20260910/`. The older `submission/` directory is
historical. Grand Challenge currently links to
[`clear-ec-algo`](https://github.com/leebyounghan/clear-ec-algo) according to the team;
that repository has NOT been synchronized by this research-repository update.
The September 10 PDF and review package remain historical snapshots, not updated
documentation of the September 14 SSL weights.

> **Continuing on a new GPU?** Read [`docs/REPRODUCE.md`](docs/REPRODUCE.md) first (env + data + how to
> retrain), then [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) (which large files to copy over), then
> [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) (what to work on).

---

## Where we stand

| Submission | Official Phase I score | Status |
|---|---:|---|
| **Lamda&KIM — September 14 submission** | **[8.7010](https://clear-ec.grand-challenge.org/evaluation/7606a2b6-818f-46d0-a1fc-d060648fc2c6/)** | Latest team-reported result; rank 2 in supplied row; improvement 0.0032 over September 10 |
| **Lamda — September 10 submission** | **[8.7042](https://clear-ec.grand-challenge.org/evaluation/0a82fe58-7f9a-4a88-b744-bcfa84b1e43a/)** | Succeeded; **rank 1** when checked September 12, 2026 |
| Earlier Lamda submission | [9.0009](https://clear-ec.grand-challenge.org/evaluation/84234081-8f49-4142-8440-9790f5de11bb/) | Historical result; not the current model |

Metric = mean of the three per-metric MAPEs (%); lower is better. Rank can change.
The linked September 10 page reports evaluation ID `0a82fe58-7f9a-4a88-b744-bcfa84b1e43a`,
image version `730ff3f7-ea96-4180-936d-bc02a4eca762` and model version
`db061942-478a-42b6-a6d7-b6dff1626108`. Per-metric official scores are not displayed
on that page. Internal validation scores, calibration estimates and later research
experiments are not official challenge results. No proven performance ceiling is asserted.

## September 10 baseline methodology

A **Cellpose-SAM encoder with shared features and three regression heads** predicts
CD, CV and HEX in standardized log space. The final fits use **8,998 eligible images**
from the 9,000-image training archive, with slide-averaged positive labels. The
submitted ensemble has **six equally weighted models**: three central-crop seeds,
one mixed-loss central model, one two-crop central model and one quality-selected
full-native-field model. Each member uses five native256 crops with eight dihedral
transforms (40 views), then decoded member predictions are averaged and fixed
per-metric calibration is applied. Inference is FP16. Later domain-SSL and raw-label
experiments are not part of this release. See the
[technical report](review_submission_20260910/output/pdf/Technical_Report.pdf) and
[historical training recipes](review_submission_20260910/TRAINING.md).

## Repo layout

```
review_submission_20260910/  frozen six-member inference, training recipes and technical PDF
code/            historical training + eval + analysis scripts
submission/      older Grand Challenge container; not the September 10 release
docs/
  METHODOLOGY.md   the pipeline, metric definitions, model/ensemble design
  FINDINGS.md      historical experiments and hypotheses; not proof of a noise floor
  REPRODUCE.md     env setup, data regeneration, exact training commands
  ARTIFACTS.md     manifest of large files (checkpoints, caches) + how to transfer/regenerate
  NEXT_STEPS.md    prioritized roadmap for the next GPU
results/         small result JSONs (per-run val/test MAPEs) + norm_consts.json
env/             environment.yml / requirements-clearec.txt
```

## Latest prepared release and historical baseline

Latest model-only archive: `model_full_quality77_ssl_raw_20260914.tar.gz`
(7,964,665,157 bytes), SHA256
`e37c83db830cc5fa5215f7fd63267beb11e6c09a8778aefe2b8c6180a37a9730`.
It preserves the first five models and calibration, and replaces the sixth with
standard custom masked-EMA SSL (all 9,000 images, 6,000 updates) followed by
raw-label regression (8,998 eligible images, 5,215 updates). This is not DINO or
I-JEPA. See [release provenance and limitations](docs/SSL_RELEASE_20260914.md).
Weights are not hosted in Git; local delivery is not evaluator-accessible hosting.

The September 10 release uses **separate container and model archives**:

- Container: `clear_ec_full_quality77_20260909.tar.gz` (approximately4.06GB).
- Model: `model_full_quality77_20260909.tar.gz` (approximately7.96GB), mounted at `/opt/ml/model`.
- Configuration: `candidate6_full_quality.json`, six members, FP16 inference.

Use the [review-package instructions](review_submission_20260910/README.md) and
[technical report PDF](review_submission_20260910/output/pdf/Technical_Report.pdf).
The old v4 single-archive instructions do not reproduce this submission.
