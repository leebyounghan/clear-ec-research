---
id: idea-ensemble-distillation
type: idea
name: Distil the multi-family ensemble into one model
summary: A student trained on the ensemble's soft targets keeps the ensemble's diversity in one deployable model.
status: closed
origin: claude
aims_at:
- all
owner_lane: backbone
cost: ~12 GPU-h per metric model
cost_hours: 12
cost_unit: GPU-h
potential_metric: all
potential_gain: null
potential_basis: 'deployability (12 h rule), not an OOF gain: a student at best matches the ensemble'
source: 'docs/LANES_20261108.md ''Facts per lane: Backbone'' (never tried)'
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-other-families-err-differently]]'
builds_on:
- '[[method-nested-blend]]'
tested_by:
- target: '[[exp-bb-ensemble-distillation]]'
  closeness: same
  note: 'folds 0+4: 9.001 vs plain DINOv3-B 9.2485, 44 % of the gap to the 11-10 rehearsal candidate (8.6876); below the half-gap rule; stack weight 0 (L1)'
- target: '[[exp-bb-checkpoint-average]]'
  closeness: related
  note: recipe A seeds are near-identical, so distilling seeds is pointless; families are not
---
# Distil the multi-family ensemble into one model

## Hypothesis

A student trained on the ensemble's soft targets keeps the ensemble's diversity in one deployable model.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Closed by the backbone lane; only '44 % of the gap' is recorded. OOF teachers make a student's fold read optimistic by about 0.08-0.13 MAPE per metric (cell-route lane).

## Update 2026-11-13

The gap is now recorded: 9.001 against 9.2485 on folds 0+4, 44 % of the way to the 11-10 integration-rehearsal candidate (8.6876), which is not the frozen candidate (O-BB-SYNFT3-5F-CORR).
