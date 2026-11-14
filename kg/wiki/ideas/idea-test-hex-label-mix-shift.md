---
id: idea-test-hex-label-mix-shift
type: idea
name: A HEX label-mix shift in the test set as the explanation of the HEX test gap
summary: A shift of the test set's HEX label mix is the one covariate shift that fits submission 1's test pattern plausibly (KL 0.006; HEX >= 0.66 x1.27); it cannot be observed without test labels.
status: open
origin: claude
aims_at:
- HEX
owner_lane: cv
cost: not testable without test labels
potential_metric: HEX
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: docs/lane_summaries_20261113/final_summary_cv_20261113.md section 4c (idea-test-hex-label-mix-shift); docs/FINAL_SUMMARY_20261113.md section 3.3; lane_cv_20261108/reads/hex_test_gap_20261111.{md,json}
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[find-test-gap-within-sampling-noise]]'
- '[[find-hex-test-gap]]'
- '[[exp-cv-hex-test-gap-composition]]'
tested_by:
- target: '[[exp-cv-hex-test-gap-composition]]'
  closeness: related
  note: the gap is inside 1000-image sampling noise; an offset fix has negative expected value with unknown sign (+0.35 to +0.42)
---

# A HEX label-mix shift in the test set as the explanation of the HEX test gap

A shift of the test set's HEX label mix is the one covariate shift that fits submission 1's test pattern plausibly (KL 0.006; HEX >= 0.66 x1.27); it cannot be observed without test labels.

## Hypothesis

If the hidden test holds more high-HEX frames than the training folds, the HEX MAPE moves without any model fault.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Risks

No HEX adjustment is safe ([[find-test-gap-within-sampling-noise]]); the HEX lane tested no test-time adjustment.

## Outcome 2026-11-13

Open and unobservable. A separate bias analysis report (`analysis_bias_20261113.md`) was still being written when this node was created and is not ingested.
