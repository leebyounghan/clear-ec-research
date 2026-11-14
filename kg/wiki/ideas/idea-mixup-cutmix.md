---
id: idea-mixup-cutmix
type: idea
name: Mixup or cutmix under recipe A
summary: Mixing images and labels regularises the regressor beyond flips.
status: closed
origin: claude
aims_at:
- all
owner_lane: backbone
cost: ~4 GPU-h (DINOv3 or recipe A, folds 0,4)
cost_hours: 4
cost_unit: GPU-h
potential_metric: all
potential_gain: 0.002
potential_basis: 'estimate: C-Mixup gave ~3 % relative MAPE on echo EF; mixing two cell images does not preserve CD/CV/HEX, so only label-distance partner mixing is sound'
source: 'docs/LANES_20261108.md ''Facts per lane: Backbone'' (never tried under recipe A)'
novelty_check: 'KG check 2026-11-09: no experiment mixes images; paper yao2022cmixup (C-Mixup, partners chosen by label distance) reports small consistent RMSE/MAPE gains in regression. Web search not yet done.'
motivated_by:
- '[[find-recipe-a-changes-converge]]'
- '[[exp-hist-encoder-lr-x5]]'
builds_on:
- '[[method-recipe-a]]'
tested_by:
- target: '[[exp-bb-cmixup-dinov3]]'
  closeness: same
  note: 'C-Mixup DINOv3-B: member 9.233 vs 9.2485 on folds 0+4; stack weight 0 in CD, CV and HEX'
- target: '[[exp-hist-augmentation-bundle]]'
  closeness: related
  note: photometric bundle, not mixing
literature:
- target: '[[yao2022cmixup]]'
  relation: suggests
- target: '[[kim2024confrag]]'
  relation: explains
  note: C-Mixup is the strongest non-ConFrag baseline under label noise
---
# Mixup or cutmix under recipe A

## Hypothesis

Mixing images and labels regularises the regressor beyond flips.

## Notes

Mixing two endothelium images changes cell density and shape statistics; labels would need mixing in the right space (CD in area, CV/HEX approximately).

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Closed: C-Mixup (the label-distance version) got stack weight 0 in all three lanes ([[find-better-alone-not-stack-weight]]).

## Update 2026-11-13

Stack reads on record: CD +0.0099 added, CV +0.0073, HEX +0.0153 on folds 0+4 in the chain and +0.00108 in the ridge ([[exp-bb-cmixup-dinov3]]).
