---
id: con-screen-folds-0-4
type: constraint
name: Screen on folds 0 and 4 against the best pipeline; extend kept changes to five folds
summary: Screen on folds 0 and 4 (3581 rows) against the best pipeline on the same rows (mean 8.7578); keep a change that lowers your metric there without hurting the other two; extend kept changes to five folds before the freeze.
status: active
origin: claude
date: '2026-11-08'
scope: global
source: docs/LANES_20261108.md 'Reading results'
---

# Screen on folds 0 and 4 against the best pipeline; extend kept changes to five folds

## Rule

Screen on folds 0 and 4 (3581 rows) against the best pipeline on the same rows (mean 8.7578); keep a change that lowers your metric there without hurting the other two; extend kept changes to five folds before the freeze.

## Notes

Folds 0 and 4 overstated several members (ConvNeXt on CD, DINOv3 on CV, 50-epoch graphs): see find-screen-folds-overstate.
