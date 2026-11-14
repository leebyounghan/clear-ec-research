---
id: con-no-statistical-gates
type: constraint
name: No CI gates or minimum effect sizes
summary: 'Keep any change that lowers the metric on OOF and stack small gains; one point estimate per try; CIs are information only. PI 2026-11-08: the goal is a higher score, not a proven finding.'
status: active
origin: user
date: '2026-11-08'
source: docs/LANES_20261108.md 'Reading results'; memory/clear-ec-no-strict-statistical-gates.md
constrains:
- '[[dec-001-plan-20261106]]'
---

# No CI gates or minimum effect sizes

## Rule

Keep any change that lowers the metric on OOF and stack small gains; one point estimate per try; CIs are information only. PI 2026-11-08: the goal is a higher score, not a proven finding.

## Notes

Supersedes the gate parts of the 11-06 plan (two-fold mean <= -0.05 with CI below 0; added <= -0.02; corrections <= -0.01 with 4 of 5 folds). Several earlier closures were made under those gates (DINOv3 B/16 at -0.0166, invariance training at -0.042).
