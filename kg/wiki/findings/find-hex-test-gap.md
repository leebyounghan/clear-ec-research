---
id: find-hex-test-gap
type: finding
name: Submission 1 HEX on test was 0.40 worse than OOF (94th percentile of OOF draws)
summary: 'Test minus OOF: CD -0.062, CV -0.061, HEX +0.395; among random 1000-image donor-grouped OOF draws the test HEX sits at percentile 94.'
status: robust
metrics:
- HEX
date: '2026-11-08'
source: docs/BRIEF.md 'Our position', 'How far OOF can be trusted'
literature:
- target: '[[zhang2020bigaug]]'
  relation: suggests
  note: stacked appearance augmentation against unseen-domain shift, free at inference
- target: '[[adachi2025ssa]]'
  relation: suggests
  note: regression TTA by subspace alignment; BN-adapt can hurt
---
# Submission 1 HEX on test was 0.40 worse than OOF (94th percentile of OOF draws)

Test minus OOF: CD -0.062, CV -0.061, HEX +0.395; among random 1000-image donor-grouped OOF draws the test HEX sits at percentile 94.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
