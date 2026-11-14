---
id: idea-test-time-adaptation-regression
type: idea
name: Source-free test-time adaptation for regression
summary: Align test-image features to the training feature subspace (normalisation-layer updates) to cut shifts such as the HEX test gap.
status: dropped
origin: literature
aims_at:
- HEX
- all
owner_lane: hex
cost: ~2 GPU-h + a PI call on test-batch statistics
cost_hours: 2
cost_unit: GPU-h
potential_metric: HEX
potential_gain: 0.01
potential_basis: 'speculative estimate: aims at the +0.395 HEX test-minus-OOF gap, which OOF cannot measure; BN-adapt hurt regression in adachi2025ssa; per-batch test statistics need a PI decision'
source: paper notes adachi2025ssa, kiet2026psc, wang2021tent
novelty_check: 'KG check 2026-11-09 (kg_query check adaptation; check test-time): no experiment adapts at test time; flip TTA is averaging only. Web search not yet done.'
motivated_by:
- '[[find-hex-test-gap]]'
- '[[find-recent-images-darker]]'
builds_on:
- '[[method-recipe-a]]'
tested_by:
- target: '[[exp-bb-leave-one-source-out]]'
  closeness: related
  note: the shift is 85-90 % a flat level offset; test-time alignment was not started
- target: '[[exp-cv-hex-test-gap-composition]]'
  closeness: related
  note: no HEX adjustment is safe when the sign of the offset is unknown
- target: '[[exp-cd-flip-tta-snapshot]]'
  closeness: related
  note: test-time averaging, not adaptation
literature:
- target: '[[adachi2025ssa]]'
  relation: suggests
- target: '[[kiet2026psc]]'
  relation: suggests
- target: '[[wang2021tent]]'
  relation: suggests
- target: '[[karani2021tta]]'
  relation: suggests
  note: adapt only an input-normalisation module (per-image cost too high as published)
- target: '[[zhang2020bigaug]]'
  relation: explains
  note: training-time augmentation is the cheaper alternative
---
# Source-free test-time adaptation for regression

## Hypothesis

Align test-image features to the training feature subspace (normalisation-layer updates) to cut shifts such as the HEX test gap.

## Notes

Any adaptation must not key on source or ID (con-no-source-id-shortcuts); a per-batch statistic of the 1000 test images is a different thing and needs a PI call.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Not started (backbone lane notes 2026-11-11 05:15 EST; F-SHIFT-001: 'test-time alignment not started'). F-TESTGAP-001: an offset fix has negative expected value when its sign is unknown.

## Update 2026-11-13

Not run at close-out (backbone lane final summary, idea-test-time-subspace-alignment: 'the coordinator's view was that a level offset cannot be fixed without labels'). The HEX lane tested no HEX test-time adjustment either.
