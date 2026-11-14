---
id: idea-label-fraction-curve
type: idea
name: Label-fraction learning curve (plan G6)
summary: Whether more data would help decides the value of external data.
status: partially-tested
origin: claude
aims_at:
- all
owner_lane: coordinator
cost: ~4 GPU-h (25 % fraction, fold 0)
cost_hours: 4
cost_unit: GPU-h
potential_metric: all
potential_gain: null
potential_basis: a measurement that values external data, not an OOF gain
source: docs/PLAN_20261106.md row G6; docs/ASSUMPTION_AUDIT_20261106.md belief 6
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-label-fraction-cost]]'
builds_on:
- '[[method-recipe-a]]'
tested_by:
- target: '[[exp-gpu0-label-fraction]]'
  closeness: same
  note: 50 % on fold 0 costs ~0.08; 25 % unread
---
# Label-fraction learning curve (plan G6)

## Hypothesis

Whether more data would help decides the value of external data.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
