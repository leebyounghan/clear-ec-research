# Submission-1 package — hybrid (Phase-1 six + recipe A ×3), built 2026-10-30

Location (build box `qudgks@210.107.192.239`): `~/workspace/clear-ec/package_20261030/`

| File | Size | SHA256 |
|---|---:|---|
| `clear-ec-hybrid_20261030.tar.gz` (docker save, gzip) | 4.26 GB | `2301d972ef65bfb5f1b7aeb77ee764e2854c46b95a31663df9d6c6132fbc0b61` |
| `model_hybrid_20261030.tar.gz` (mount at `/opt/ml/model`) | 4.80 GB | `1134b453653b4a7215532be0be9dddc5f2a29d66af02cdaf4a0f7dbae506bf87` |

Image: `clear-ec-hybrid:20261030` (7.87 GB uncompressed), base `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime`
(same as the Phase-1 submission; runs on sm75/T4), plus `timm==1.0.20` and the vendored `dinov3` runtime.
Build context: `submission_hybrid_20261030/` in this repository (Dockerfile, frozen sources, entrypoint).
Entrypoint: `python /opt/app/hybrid_entrypoint.py --config candidate6_full_quality.json --fp16`
(Grand Challenge I/O layout unchanged from Phase 1: `/input/inputs.json`, one `.mha`, three JSON outputs).

Model mount contents (`MODEL_SHA256SUMS.txt`): the Phase-1 six members + `cpsam_v2` + `candidate6_full_quality.json`
(byte-identical to the 2026-09-14 release where the release manifest records them), `recipea/member_s{46,47,48}.pt`
(all-data recipe A, `experiments/fulldata_recipeA_20261028/regression_std_s*/runs/fold99/cpdino_vitb/epoch_20.pt`),
`cpdino/cpdino-vitb` (CellposeDINO ViT-B backbone, sha 3ed4c06a…).

## What the container computes
`output = 0.5 · Phase1(image) + 0.5 · RecipeA(image)`
- `Phase1`: the unchanged Phase-1 app — six crop-based CPSAM members, 40 views each, equal mean, fixed scalars
  (CD 0.99704, CV 0.98797, HEX 0.98922).
- `RecipeA`: three whole-frame CPDINO members (std pooling); per member, four H/V-flip views averaged in
  standardized-log space and decoded with the member's training log-mean/std; equal mean over members; per-target
  scalars fitted on the five-fold OOF (CD 0.99085, CV 0.98691, HEX 0.99296).
- fp16 inference; the entrypoint asserts PyTorch's memory-efficient SDPA kernel for the 20,336-token attention
  before reading the input (the math-only kernel would need 18.5 GiB and is refused).

## Verification performed
- Pod integration test (`results/submission1_test_20261030/n60_result.json`, 60 fold-4 images): recipe A path
  reproduces stored TTA predictions to 0.15 %; three members agree within 0.5–0.8 % (log spread); peak GPU memory
  2.64 GiB under a 15 GiB allocator cap; 12.9 s/image on an H200 (≈11 s of it is the Phase-1 240 views).
- Container smoke test on the build box (RTX 4090, `--network none --memory 16g`, model mounted read-only),
  image 0234-20: CD 2977.94 / CV 0.41633 / HEX 0.48083 — identical to the pod test to four decimals; 22 s wall.
- Phase-1 app outputs differ from the stored fold-clean OOF by a median 1.1 % (one image 17 %): expected, because
  the release members are all-data refits and fold-4 images are in their training set; precision (fp16/bf16/fp32)
  contributes 0.0 %.

## Evidence for the composition
Five-fold OOF (8,954 images): hybrid 8.877 vs recipe A 8.951 vs Phase-1 five 9.002 (hybrid better on 5/5 folds,
95 % CI of the difference vs recipe A [−0.121, −0.044]). Fold-4 matched 1,778 rows vs the submitted Phase-1 six:
8.798 vs 8.888 (−0.090; CI [−0.149, −0.015]; P(better) 0.99). Files: `results/recipeA_analysis_20261028/`.
Pre-registration: `results/recipe_A_20261027/preregistration_20261027.json` + amendment 2026-10-28; decisions
`decision_20261029_submission1.json`, `decision_20261030_pi.json`.

## Limitations to state on the platform
Development folds were reused for recipe selection (OOF is optimistic); patient-level independence of the folds is
unverified; the Phase-1 members' OOF used validation-selected checkpoints; no T4 timing was measured (H200/4090 only).

## Upload (PI, window 2026-11-05 … 11-07)
Upload the two archives; verify with `sha256sum --check SHA256SUMS.txt`; keep the exact uploaded files and the
platform receipt under `results/submission1_package_20261030/`.
