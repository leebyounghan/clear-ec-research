---
id: find-test-gap-within-sampling-noise
type: finding
name: Submission 1's test-minus-OOF pattern is inside 1000-image sampling noise (F-TESTGAP-001)
summary: 'Test minus OOF (CD -0.062, CV -0.061, HEX +0.395) is inside 1000-image sampling noise: joint tail 0.40, HEX alone 6.4 %; SD of the 1000-row mean gap CD 0.318, CV 0.518, HEX 0.257. No HEX adjustment is safe: an offset fix has negative expected value when its sign is unknown.'
status: robust
metrics:
- CD
- CV
- HEX
date: '2026-11-11'
source: shared notes ledger_20261111.jsonl F-TESTGAP-001; lane_cv_20261108/reads/hex_test_gap_20261111.md
qualifies:
- target: '[[find-hex-test-gap]]'
  note: the +0.395 HEX gap (94th percentile of OOF draws) is within sampling noise once the three metrics are read jointly (joint tail 0.40)
evidence_against:
- target: '[[idea-appearance-aug-hex-gap]]'
  note: the gap it aims at is inside sampling noise
- target: '[[idea-test-time-adaptation-regression]]'
  note: no HEX adjustment is safe when the sign of the offset is unknown
---

# Submission 1's test-minus-OOF pattern is inside 1000-image sampling noise (F-TESTGAP-001)

Test minus OOF (CD -0.062, CV -0.061, HEX +0.395) is inside 1000-image sampling noise: joint tail 0.40, HEX alone 6.4 %; SD of the 1000-row mean gap CD 0.318, CV 0.518, HEX 0.257. No HEX adjustment is safe: an offset fix has negative expected value when its sign is unknown.

## Evidence

- [[exp-cv-hex-test-gap-composition]].

## Limits

One submission, 1000 test rows; the test labels are hidden.

## Bears on

Closes HEX-specific corrections aimed at the gap: [[idea-hex-test-gap-composition]], [[idea-appearance-aug-hex-gap]], [[idea-test-time-adaptation-regression]].
