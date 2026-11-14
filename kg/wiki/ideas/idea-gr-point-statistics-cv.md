---
id: idea-gr-point-statistics-cv
type: idea
name: Point statistics robust to misses, e.g. g(r), for CV (plan C3)
summary: Pair-correlation statistics degrade less under detector misses than area CV.
status: open
origin: claude
aims_at:
- CV
owner_lane: cv
cost: ~2 CPU-h (features from stored TTA peaks + nested stack read)
cost_hours: 2
cost_unit: CPU-h
potential_metric: CV
potential_gain: 0.002
potential_basis: 'estimate: detector-point statistics carry little beyond the stack (Konan k=2 from peaks r 0.028 with the residual); neighbour-distance CV degrades less under misses, but the graph network already reads the peaks'
source: docs/PLAN_20261106.md row C3
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-detector-misses-ruin-area-cv]]'
builds_on:
- '[[method-pre-only-detector]]'
tested_by:
- target: '[[exp-hist-simple-cv-estimators]]'
  closeness: related
  note: simple estimators closed; numbers not recorded
literature:
- target: '[[piorkowski2017segmentation]]'
  relation: suggests
  note: CV of side length is stable across segmentations
- target: '[[viguerasguillen2020dsaek]]'
  relation: explains
  note: area CV is overestimated below ~50 cells
---
# Point statistics robust to misses, e.g. g(r), for CV (plan C3)

## Hypothesis

Pair-correlation statistics degrade less under detector misses than area CV.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
