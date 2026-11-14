---
id: find-two-fold-ci-width
type: finding
name: Two-fold paired reads have CI about +-0.055; five folds +-0.035
summary: Same-seed paired delta SD 0.04; different seeds 0.074. A true -0.05 clears zero 42 % of the time on two folds, 80 % on five.
status: robust
metrics:
- mean
date: '2026-11-08'
source: memory/clear-ec-mlops-review-20261106.md
qualifies:
- '[[find-screen-folds-overstate]]'
---

# Two-fold paired reads have CI about +-0.055; five folds +-0.035

Same-seed paired delta SD 0.04; different seeds 0.074. A true -0.05 clears zero 42 % of the time on two folds, 80 % on five.

## Limits and notes

Information only: the PI dropped CI gates on 11-08 (con-no-statistical-gates).

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
