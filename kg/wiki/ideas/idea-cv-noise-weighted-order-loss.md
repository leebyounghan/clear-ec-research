---
id: idea-cv-noise-weighted-order-loss
type: idea
name: Noise-weighted pairwise order loss for CV
summary: Add a pairwise order loss on the CV head where each pair is weighted by the probability that label i truly exceeds label j under the known label noise, so near-ties carry no gradient.
status: adopted
origin: literature
aims_at:
- CV
owner_lane: cv
cost: ~1 GPU-h on frozen features (five folds); ~4 GPU-h end to end on folds 0,4
cost_hours: 1
cost_unit: GPU-h
potential_metric: CV
potential_gain: 0.005
potential_basis: 'estimate: frozen CV-tilted losses read +0.01 to +0.04 (worse) and the stack gave them weight 0; order learning gains in the paper are ~2.5 % relative MAE on rank tasks, and our score is MAPE, not rank'
source: Literature review 2026-11-09 (paper notes' suggests_try), coordinator list of eight suggestions, suggestion 7
novelty_check: 'KG check 2026-11-09 (kg_query check + grep over wiki/): check order loss; check rank; check cv loss -> exp-cv-frozen-heads-cv-loss (closed), exp-cv-dinov3-cv-weighted-loss (closed), idea-imbalanced-regression-cv-tails (RankSim, never tried). Literature: papers from the KG literature review (lee2026stochorder, gong2022ranksim); no new web search in this pass.'
motivated_by:
- '[[find-cv-label-discrete]]'
- '[[find-sampling-se-150-cells]]'
- '[[find-lit-cv-hex-grader-noise]]'
builds_on:
- '[[method-frozen-token-heads]]'
tested_by:
- target: '[[exp-cv-order-loss-graph-quartet]]'
  closeness: same
  note: 'on the graph quartet, as its own stack component: CV 10.1101 vs best pipeline 10.1512 (-0.0411), five folds; kept'
- target: '[[exp-cv-order-loss-dinov3]]'
  closeness: close
  note: 'on DINOv3: -0.0460 with seed 46, -0.0369 with 3 seeds; not adopted'
- target: '[[exp-cv-order-loss-other-members]]'
  closeness: close
  note: 'on ViT-L, DINOv2 and other graph sets: zero or worse; numbers not recorded'
- target: '[[exp-cv-frozen-heads-cv-loss]]'
  closeness: related
  note: 'CV-weighted / CV-only losses on the same frozen features: +0.01 to +0.04, stack weight 0'
- target: '[[exp-cv-dinov3-cv-weighted-loss]]'
  closeness: related
  note: metric weighting end to end, no gain
literature:
- target: '[[lee2026stochorder]]'
  relation: source-of-method
  note: probabilistic pairwise margin loss under known label noise
- target: '[[gong2022ranksim]]'
  relation: suggests
  note: rank regulariser; needs backbone gradients
---
# Noise-weighted pairwise order loss for CV

## Hypothesis

With ~6.5 % (or more) CV label noise, MSE/L1 pulls predictions to the mean; an order loss that ignores near-ties keeps the far pairs ordered and widens the prediction spread where it is real.

## Closest tries in the KG

- [[exp-cv-frozen-heads-cv-loss]], [[exp-cv-dinov3-cv-weighted-loss]] (related: loss changes on the CV head, no gain).

## Literature

lee2026stochorder, gong2022ranksim.

## Minimal test

Frozen recipe A token features, five folds, CV head with L1 + noise-weighted order loss (sigma = max(sampling SE, fitted)); add as a CV correction to the lane best (10.114), nested.

## Risks

Compression is expected at r 0.686 ([[find-lit-cv-compression-expected]]); a wider spread can raise MAPE.

## Outcome 2026-11-11

Adopted on the graph quartet (sixth component of D-CV-001). The order-loss variants on image members are closed.
