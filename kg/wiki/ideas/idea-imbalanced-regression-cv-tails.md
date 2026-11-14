---
id: idea-imbalanced-regression-cv-tails
type: idea
name: Label-distribution smoothing for CV tails
summary: Inverse-density loss weights cut tail error, where a third of the CV error sits.
status: closed
origin: literature
aims_at:
- CV
owner_lane: cv
cost: ~1 GPU-h frozen / ~4 GPU-h end to end (folds 0,4)
cost_hours: 1
cost_unit: GPU-h
potential_metric: CV
potential_gain: 0.002
potential_basis: 'estimate: premise contradicted (find-lit-cv-compression-expected, find-lit-cv-hex-grader-noise); LDS overall gain ~3 % relative in the paper with central error up'
source: paper note yang2021imbalanced
novelty_check: 'KG check 2026-11-09: tail pooling and CV-weighted losses are closed; density-based reweighting of CV labels has not been tried.'
motivated_by:
- '[[find-cv-error-in-tails]]'
builds_on:
- '[[method-metric-weighted-loss]]'
tested_by:
- target: '[[exp-cv-lds-member]]'
  closeness: same
  note: 'LDS member on top of the CV stack: zero or worse; number not recorded'
- target: '[[exp-hist-tail-pooling]]'
  closeness: related
- target: '[[exp-cv-dinov3-cv-weighted-loss]]'
  closeness: related
  note: weights the metric, not the label density
literature:
- target: '[[yang2021imbalanced]]'
  relation: suggests
- target: '[[gong2022ranksim]]'
  relation: suggests
- target: '[[vidalvillegas2024eyebank]]'
  relation: contradicts
  note: 'CV/HEX labels mostly grader noise: prioritise decoding/shrinkage over fitting outer deciles'
- target: '[[delange2022mindgap]]'
  relation: contradicts
  note: compressed tails are expected at r 0.686
- target: '[[desanctis2006twodevices]]'
  relation: contradicts
  note: reader ceiling for CV/HEX (ICC < 0.65)
---
# Label-distribution smoothing for CV tails

## Hypothesis

Inverse-density loss weights cut tail error, where a third of the CV error sits.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Closed: the LDS member read zero or worse on the CV stack (CV lane notes 2026-11-11 03:15 EST); the value is not recorded.

## Update 2026-11-13

Number now on record: the LDS member got weight 0 on folds 0+4 ([[exp-cv-lds-member]]).
