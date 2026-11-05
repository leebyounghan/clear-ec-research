# T4-container numerics check of the hex6 correction (2026-11-01)

**Verdict: the shipped hex6 component stays as is.** Both conditions of the parent's pre-registered rule pass.

| condition (parent rule) | limit | measured | pass |
|---|---|---|---|
| seven-family LOFO delta under fp16 vs bf16, same checkpoints | within 0.005 pp | bf16 -0.0291 vs fp16 -0.0281 pp (+0.0010) | yes |
| deployed a/b/w applied unchanged to fp16 features, all fold-clean rows | not worse than -0.025 pp | -0.0326 pp mean (5/5 folds improving) | yes |

## What was run

* T4 numerics = fp16 autocast + `sdpa_kernel([EFFICIENT_ATTENTION, MATH])` around the encoder forward; heads in float32; peak finding, NMS, Voronoi and hex6 unchanged. The reference numbers are the original bf16 (H200, flash SDPA) OOF outputs.
* Fold-clean OOF re-extraction on lane01 (GPUs 0-1), 35 family x fold items, 13:15-14:49 UTC; CPU analysis 14:49-15:23 UTC. Base = hybrid B (`recipeA_analysis_20261028/hybrid_oof.npz`, 8.8772 on 8954 rows), LOFO log-blend with t=2 (HEX).
* Seven of the ten ACTIVE families have fold-clean checkpoints (s8_s46, sg2_s46..s50, sg2pre_s46). s8_s47, s8_s48 and s8multi_s47 were never saved; per the parent's decision they were **not retrained**, their bf16 OOF feature is kept in the ten-family blends, and the ten-family numbers are **context only**.

## Numbers (pp on the official mean MAPE, base B, folds 0-4)

| blend | mean | HEX | per fold | LOFO w per fold |
|---|---|---|---|---|
| seven bf16 (decisive reference) | -0.0291 | -0.0874 | -0.0440 -0.0320 -0.0398 -0.0244 -0.0055 | 0.26 0.26 0.26 0.28 0.32 |
| seven fp16 (decisive) | -0.0281 | -0.0844 | -0.0412 -0.0265 -0.0414 -0.0258 -0.0058 | 0.24 0.28 0.26 0.28 0.30 |
| ten bf16 (shipped value) | -0.0311 | -0.0934 | -0.0473 -0.0309 -0.0415 -0.0274 -0.0086 | 0.26 0.28 0.28 0.28 0.32 |
| ten mixed, fp16 where available (context, flagged) | -0.0297 | -0.0891 | -0.0442 -0.0252 -0.0418 -0.0278 -0.0095 | 0.24 0.26 0.26 0.28 0.30 |
| deployed a/b/w on ten bf16 (in-sample) | -0.0331 | -0.0994 | -0.0508 -0.0309 -0.0419 -0.0279 -0.0143 | fixed 0.28 |
| deployed a/b/w on ten mixed fp16 (in-sample) | -0.0326 | -0.0979 | -0.0519 -0.0265 -0.0429 -0.0293 -0.0128 | fixed 0.28 |

Deployed parameters: a = -0.2084, b = 0.5989, w = 0.28 (`deploy/hex6_blend_params.json`). The LOFO-chosen w under fp16 stays in the same 0.24-0.30 band.

## Feature stability bf16 vs fp16 (log hex6)

| level | r | mean abs diff | rows changed |
|---|---|---|---|
| per family s8_s46 / sg2_s46..s50 | 0.90-0.92 | 0.042-0.053 | 96-99% |
| per family sg2pre_s46 (pre-trained only) | 0.959 | 0.028 | 99% |
| seven-family mean | 0.980 | 0.019 | - |
| ten-family mean (mixed) | 0.989 | 0.013 (max 0.52 on one row) | - |

Reading: fp16 moves individual detectors' peak sets a lot (per-detector r about 0.9, about the seed-to-seed level), but the ensemble mean is stable (r 0.98-0.99) and the blend gain shrinks by only about 0.001 pp. The deployed a/b/w transfer unchanged.

## Deployment gap found while checking (handled by the parent outside this module)

* `deploy/hex6_correction.py` hard-codes `torch.autocast('cuda', dtype=torch.bfloat16)` and does not restrict SDPA backends; the 25-image fixture is from the H200 bf16 run. Escalated 15:28 UTC.
* Parent's answer (15:35 UTC): the submission-2 container uses `submission2_test_20261101/source/hex6_t4_20261101.py`, a `Hex6Corrector` subclass that overrides only `hex6_per_detector` (fp16 autocast + `sdpa_kernel([EFFICIENT_ATTENTION, MATH])`). Verified on the pod: 25-image fixture parity (`experiments/submission2_test_20261101/hex6_parity.json`) and a 60-image fold-4 integration run under a 15 GiB cap (`n60_sub2_result.json`: CD/CV unchanged, hex6 applied 60/60, deployed-vs-OOF feature r 0.967, mean offset 0.0018). The container acceptance test checks behaviour, not bf16 bit-reproduction, so no fp16 fixture is needed.
* Consequently `deploy/` stays frozen (not edited), and the per-row tail analysis (`feature_tail.py`) was stopped before completion as not needed.

## Files

`t4_numerics/result.json` (all numbers), `protocol.json` (pre-registration), `analyze.log`, `extract_gpu{0,1}.log`, `run_t4_extract.sh`, `source/green_t4_numerics_20261101.py`, fp16 outputs under `t4_numerics/fp16/<family>/fold<k>/`.
