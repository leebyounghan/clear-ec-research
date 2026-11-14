---
id: idea-pseudo-label-student-e11
type: idea
name: No-label student training (E11)
summary: A student trained on pseudo-labels without human labels (E11a student, E11b mixed).
status: open
origin: claude
aims_at:
- all
owner_lane: backbone
cost: ~12 GPU-h
cost_hours: 12
cost_unit: GPU-h
potential_metric: all
potential_gain: 0.001
potential_basis: 'estimate: the label-agreeing pseudo-label pool is easy frames (find-pseudo-label-pool-easy-frames)'
source: memory/clear-ec-pod-oom-two-trainings.md
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-label-fraction-cost]]'
builds_on:
- '[[method-recipe-a]]'
---
# No-label student training (E11)

## Hypothesis

A student trained on pseudo-labels without human labels (E11a student, E11b mixed).

## Notes

Ran on 2026-11-03/04; the pod was OOM-killed during it. No result in the sources read: ask before treating it as tried.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
