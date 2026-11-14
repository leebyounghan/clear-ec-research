---
id: find-image-quality-not-error-driver
type: finding
name: Noise, focus and exposure do not explain the error
summary: No frame statistic reaches Spearman 0.06 with the signed log residual within source; calibrated noise and focus sit at the shuffled floor.
status: robust
metrics:
- CD
- CV
- HEX
date: '2026-11-08'
source: experiments/lane_gpu0_20261106/REPORT.md; memory/clear-ec-error-case-analysis.md
---

# Noise, focus and exposure do not explain the error

No frame statistic reaches Spearman 0.06 with the signed log residual within source; calibrated noise and focus sit at the shuffled floor.

## Limits and notes

Rules out the image-quality mechanism only, not general regularisation.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
