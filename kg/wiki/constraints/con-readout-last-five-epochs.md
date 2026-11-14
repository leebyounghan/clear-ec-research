---
id: con-readout-last-five-epochs
type: constraint
name: Read encoder training as the mean of the last five epochs on two folds, paired
summary: 'Validation mean moves 0.059 between epochs (within-run SD 0.052): read the mean of the last five epochs, at least two folds, paired with a matched reference.'
status: active
origin: claude
date: '2026-11-06'
source: docs/BRIEF.md 'Readout noise'; docs/ASSUMPTION_AUDIT_20261106.md 'Readout noise'
constrains:
- '[[method-family-recipe]]'
- '[[method-recipe-a]]'
---

# Read encoder training as the mean of the last five epochs on two folds, paired

## Rule

Validation mean moves 0.059 between epochs (within-run SD 0.052): read the mean of the last five epochs, at least two folds, paired with a matched reference.
