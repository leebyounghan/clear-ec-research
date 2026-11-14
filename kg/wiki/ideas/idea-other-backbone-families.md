---
id: idea-other-backbone-families
type: idea
name: Swin, ResNet, EfficientNet or RETFound as members
summary: Families never tried may err differently from the ViTs.
status: never-tried
origin: claude
aims_at:
- all
owner_lane: backbone
cost: ~12 GPU-h (two folds) + PI approval for new weights
cost_hours: 12
cost_unit: GPU-h
potential_metric: CV
potential_gain: 0.005
potential_basis: 'estimate: other families err differently (add at 12.5-30 %) but on CV ViT-L reads weight 0 on five folds and CD members are saturated'
source: docs/ASSUMPTION_AUDIT_20261106.md 'Model families' (never tried)
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-other-families-err-differently]]'
builds_on:
- '[[method-family-recipe]]'
tested_by:
- target: '[[exp-gpu3-convnext-family]]'
  closeness: related
  note: ConvNeXt, a CNN, added -0.0016 overall; CD-only it helped until ViT-L
literature:
- target: '[[pachitariu2025cellposesam]]'
  relation: explains
  note: CPSAM (ViT-L SAM encoder) is the family already tried as a HEX-only member
---
# Swin, ResNet, EfficientNet or RETFound as members

## Hypothesis

Families never tried may err differently from the ViTs.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
