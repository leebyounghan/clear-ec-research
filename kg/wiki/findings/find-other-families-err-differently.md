---
id: find-other-families-err-differently
type: finding
name: Other encoder families are worse alone but err differently (residual r 0.87-0.95), so they add at 12.5-30 %
summary: DINOv3 B/16 +0.27 alone yet -0.017 added; ViT-L +0.10 to +0.22 alone yet -0.013 added; on CD error r with B 0.87-0.91.
status: robust
metrics:
- CD
- CV
- mean
date: '2026-11-08'
source: experiments/lane_cd_20261108/REPORT.md; experiments/lane_gpu3_20261106/REPORT.md; experiments/lane_backbone_20261108/RESULTS.tsv
---

# Other encoder families are worse alone but err differently (residual r 0.87-0.95), so they add at 12.5-30 %

DINOv3 B/16 +0.27 alone yet -0.017 added; ViT-L +0.10 to +0.22 alone yet -0.013 added; on CD error r with B 0.87-0.91.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
