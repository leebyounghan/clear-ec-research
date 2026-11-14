---
id: idea-device-map-true-statistic-head
type: idea
name: Low-capacity device map separate from a true-statistic head
summary: 'Model label = g(T(x), quality) + noise: a true-statistic head T(x) supervised where cell truth exists (25 expert frames, 385 Topcon point images) and a constrained, few-parameter device map g fitted on the 8953 device labels.'
status: open
origin: literature
aims_at:
- CV
- HEX
owner_lane: backbone
cost: ~12 GPU-h (one direction box, two folds end to end); ~2 GPU-h on frozen features
cost_hours: 12
cost_unit: GPU-h
potential_metric: CV
potential_gain: 0.005
potential_basis: 'estimate: the device map itself is now known from centres (Konan k=2, scale 1.09), so g adds little; T(x) has truth on only 25 + 385 frames, and cell-level truth from detector points carried no residual signal (r 0.028)'
source: Literature review 2026-11-09 (paper notes' suggests_try), coordinator list of eight suggestions, suggestion 6
novelty_check: 'KG check 2026-11-09 (kg_query check + grep over wiki/): check device map; check true statistic; check confusion -> exp-rc-mask-emulator (calibrated device emulator as a feature), exp-cv-konan-bisectors-markers (device rule from expert centres), no two-head try. Literature: papers from the KG literature review (tanno2019confusion, gasser2015konantopcon); no new web search in this pass.'
motivated_by:
- '[[find-residual-at-oracle-floor-cv-hex]]'
- '[[find-konan-rule-explains-device-cv]]'
- '[[find-lit-topcon-konan-cv-scale]]'
builds_on:
- '[[ds-expert-markers-25]]'
- '[[ds-topcon-external-385]]'
- '[[method-konan-weighted-bisectors]]'
tested_by:
- target: '[[exp-rc-mask-emulator]]'
  closeness: related
  note: device-style statistics from masks, calibrated and added; no separate true head
- target: '[[exp-cv-konan-bisectors-markers]]'
  closeness: related
  note: 'the device map from true centres: k=2 weighted bisectors, scale 1.09'
literature:
- target: '[[tanno2019confusion]]'
  relation: source-of-method
  note: annotator confusion with a trace penalty for identifiability
- target: '[[gasser2015konantopcon]]'
  relation: explains
  note: 'Topcon points carry another CV scale: g must be device-specific'
---
# Low-capacity device map separate from a true-statistic head

## Hypothesis

Separating what the cells are from how the device reads them lets the image model learn cell statistics without absorbing device and grader noise.

## Closest tries in the KG

- [[exp-rc-mask-emulator]], [[exp-cv-konan-bisectors-markers]] (related). No try of a two-head model.

## Literature

tanno2019confusion; gasser2015konantopcon for the device-scale caveat.

## Minimal test

Frozen features first: T head on recipe A tokens supervised by Konan-rule statistics from the 25 marker frames and 385 Topcon point images (k=2, Topcon scale handled by its own g), g = affine in log space with a brightness term fitted on the 8953 labels; compare CV OOF with the direct head on folds 0,4.

## Risks

Too little truth for T (410 frames, two devices); detection quality, not the device map, is the binding constraint ([[find-konan-rule-explains-device-cv]]).
