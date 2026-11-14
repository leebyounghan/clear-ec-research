---
id: idea-exclude-hyphen-cv-from-loss
type: idea
name: Leave the B-2019 hyphen rows' CV out of the training loss
summary: Those 523 rows' CV labels look noisy or differently derived.
status: open
origin: claude
aims_at:
- CV
owner_lane: cv
cost: ~4 GPU-h (DINOv3 member, folds 0,4)
cost_hours: 4
cost_unit: GPU-h
potential_metric: CV
potential_gain: 0.002
potential_basis: 'estimate: dropping 3 % suspicious labels read +0.0003; the 523 rows are ~6 % of frames; a training-time exclusion by ID format needs a check against con-no-source-id-shortcuts'
source: experiments/lane_gpu0_20261106/REPORT.md 'Audit A1' (not tested)
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-b2019-hyphen-cv-unreadable]]'
builds_on:
- '[[method-recipe-a]]'
tested_by:
- target: '[[exp-hist-label-clean-training]]'
  closeness: related
  note: 'dropping 3 % of labels: +0.0003'
---
# Leave the B-2019 hyphen rows' CV out of the training loss

## Hypothesis

Those 523 rows' CV labels look noisy or differently derived.

## Notes

Tension with con-no-source-id-shortcuts: it keys on ID format at training time (not at test). Needs a PI call before running.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
