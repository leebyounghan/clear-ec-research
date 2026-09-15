# September 10 baseline — preserved source

This directory contains the unchanged inference application, Dockerfile, model
configuration and historical training recipes used for the September 10 baseline
(score [8.7042](https://clear-ec.grand-challenge.org/evaluation/0a82fe58-7f9a-4a88-b744-bcfa84b1e43a/)).

For the September 14 release, use the
[Docker guide](../docker_review_20260914/README.md) and
[current technical report](../output/pdf/Lamda_KIM_CLEAR_EC_Technical_Report.pdf).
Its model archive changes the sixth checkpoint while reusing this application
and the original Docker image.

## Contents

| Path | Purpose |
|---|---|
| `app/`, `Dockerfile`, `constraints.txt` | Frozen inference source and dependency pins |
| `configs/candidate6_full_quality.json` | Six-member aggregation and calibration |
| `provenance/` | September 10 identities, runtime lock and training configurations |
| [TRAINING.md](TRAINING.md) | Historical all-data fitting recipes and reproducibility limits |
| `scripts/` | Historical verification and optional venv utilities |
| [Historical PDF](output/pdf/Technical_Report.pdf) | September 10 method snapshot |

The manifests and verification scripts in this directory identify the
**September 10 weights**, not the replacement sixth checkpoint. Verify the
September 14 model using
[the current manifest](../docker_review_20260914/MODEL_SHA256SUMS.txt).

The previous detailed venv/setup notes are preserved in
[the documentation archive](../docs/archive/README.md). They are not prerequisites
for the current Docker workflow.
