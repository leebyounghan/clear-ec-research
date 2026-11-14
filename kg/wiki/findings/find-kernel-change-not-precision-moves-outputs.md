---
id: find-kernel-change-not-precision-moves-outputs
type: finding
name: Precision is not what moves the deployed outputs; the attention / convolution kernel change is (T4 path impact about +0.007 whatever the dtype)
summary: 'Against the H200 reference, fp16, mixed, all-fp32, emulated bf16 and an A10G proxy all deviate by about the same amount (HEX median 0.355-0.45 %), with a strict-score impact of +0.0053 to +0.0069. On 200 labelled rows all-fp32 reads +0.0069 against fp16''s +0.0066. Thresholded detector peaks flip under any kernel change: 0 of 420 peak sets are identical even in fp32.'
status: robust
metrics:
- CD
- CV
- HEX
- mean
date: '2026-11-13'
source: shared notes ledger_20261111.jsonl O-DEPLOY-020 (line 185), O-COORD-F32-200 (line 203), O-DEPLOY-021 (line 188), O-DEPLOY-019 (line 168), F-DEPLOY-004 (line 187); lane_cellroute_20261109/out/t4_cmp_20.json, t4_impact_20.json, t4_impact200_t1_report.json, t4_impact200_p_report.json, t4_impact200_p_f32d_report.json
evidence_for:
- '[[dec-020-t4-assumed-tier-p-t3-fallback]]'
---

# Precision is not what moves the deployed outputs; the attention / convolution kernel change is (T4 path impact about +0.007 whatever the dtype)

Against the H200 reference, fp16, mixed, all-fp32, emulated bf16 and an A10G proxy all deviate by about the same amount (HEX median 0.355-0.45 %), with a strict-score impact of +0.0053 to +0.0069. On 200 labelled rows all-fp32 reads +0.0069 against fp16's +0.0066. Thresholded detector peaks flip under any kernel change: 0 of 420 peak sets are identical even in fp32.

## Evidence

- 20 frames, tier t1 ([[exp-deploy-t4-precision-paths-20-frames]]): impact fp16 + 1 copy +0.0065, fp16 4 copies +0.0066, mix +0.0053, all-fp32 +0.0053, bf16 emulated +0.0062, A10G proxy +0.0059; no path meets the 0.05 % median rule; HEX carries about 3/4 of every impact.
- 200 labelled rows: fp16 + 1 copy +0.0066 (+0.0016 to +0.0121) on t1 and +0.0067 (+0.0017 to +0.0122) on p ([[exp-deploy-t4-default-200-rows-t1]], [[exp-deploy-t4-default-200-rows-p]]); all-fp32 +0.0069 (+0.0017 to +0.0125), paired fp32 minus fp16 +0.0065 (-0.0074 to +0.0204) ([[exp-deploy-t4-all-fp32-200-rows]]).
- Mechanism ([[exp-deploy-flops-per-frame]]): the reference was computed with cuDNN attention, which only the H200 runs; an A10G gets FlashAttention-2 and a T4 has no bf16 kernels, so every other card changes the kernel.
- The real T4 agrees with the H200 emulation of its path on 3 frames (median CD 0.001 %, CV 0.04 %, HEX 0.07 %).

## Limits

The impacts are per-frame deviations applied to the strict OOF with a random sign, not a measured test change; 200 rows cannot resolve 0.006 directly (measured -0.006, 95 % -0.040 to +0.028). All dtype comparisons are H200 emulations; the real T4 was compared on 3 frames only.

## Bears on

The GPU type is a time question, not a score question (F-DEPLOY-004): fp32 does not help and does not finish in 600 s on a real T4. The 0.05 % parity rule cannot be met off the H200. Decision: [[dec-020-t4-assumed-tier-p-t3-fallback]]. Open: [[idea-t4-precision-hex]], [[idea-cv-fp16-path-impact]].
