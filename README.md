# CLEAR-EC — Research & Training Repo

Training, evaluation, and analysis code for the [CLEAR-EC Grand Challenge](https://clear-ec.grand-challenge.org/)
(Corneal Learning for Endothelial Assessment and Review). The task: predict three clinical metrics per
specular-microscopy image — **Cell Density (CD)**, **Coefficient of Variation (CV)**, **Hexagonality (HEX)**.

This repo is the **research/training** side. The **submission container** lives in a separate repo,
[`clear-ec-algo`](https://github.com/leebyounghan/clear-ec-algo) (a mirror of `submission/` here).

> **Continuing on a new GPU?** Read [`docs/REPRODUCE.md`](docs/REPRODUCE.md) first (env + data + how to
> retrain), then [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) (which large files to copy over), then
> [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) (what to work on).

---

## Where we stand

| | CD | CV | HEX | **Overall (mean MAPE)** |
|---|---|---|---|---|
| Leaderboard #1 (external) | — | — | — | **8.7720** |
| **Our submitted** (single s2, v2 container) | — | — | — | **9.0009** (2nd place) |
| Internal held-out ensemble (measured) | 6.41 | 10.39 | 9.97 | **8.92** |
| + CV/HEX calibration | 6.41 | 10.29 | 9.92 | **8.89** |
| **v4** (all-9000 + slide-smooth + calib) | — | — | — | ~8.8 *(est; unmeasurable — see below)* |

Metric = mean of the three per-metric MAPEs (%). Lower is better. Scored on a hidden ROI per image.

## The one-paragraph summary

A **Cellpose-SAM ViT-L encoder + small per-metric regression heads**, trained on ~7000 images with
log-space targets and 40-view (C5×8-dihedral) test-time augmentation, then combined into a **per-metric
greedy ensemble**. CD is at 6.4% MAPE; **CV/HEX are stuck near ~10%** and we have *proven* (two independent
experiments) that this is a **noise floor**, not a model limitation — see [`docs/FINDINGS.md`](docs/FINDINGS.md).
The final submission model (`v4`) retrains the winning recipe on **all 9000 images** with **slide-averaged
targets** and a small **β=−1 calibration** on CV/HEX.

## Repo layout

```
code/            all training + eval + analysis scripts (65 files; see docs/REPRODUCE.md for the map)
submission/      the Grand Challenge algorithm container (Dockerfile, inference.py, requirements.txt)
docs/
  METHODOLOGY.md   the pipeline, metric definitions, model/ensemble design
  FINDINGS.md      what worked, what failed, and the CV/HEX noise-floor proof
  REPRODUCE.md     env setup, data regeneration, exact training commands
  ARTIFACTS.md     manifest of large files (checkpoints, caches) + how to transfer/regenerate
  NEXT_STEPS.md    prioritized roadmap for the next GPU
results/         small result JSONs (per-run val/test MAPEs) + norm_consts.json
env/             environment.yml / requirements-clearec.txt
```

## Current best submission

`v4` container image (weights baked in, ~8.8GB compressed). Build it with
`code/finalize_container.sh` after the four `final9000_*.pt` checkpoints exist. Upload the single
`clear-ec-algo-image-v4.tar.gz` to the Grand Challenge **Containers** tab — no separate Model upload needed.
