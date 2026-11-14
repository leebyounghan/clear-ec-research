---
id: find-cv-error-in-tails
type: finding
name: 'CV: predictions span 66 % of the label spread; the two outer deciles hold a third of the error'
summary: 'Best pipeline on CV: r 0.686, residual log SD 0.138; error by label decile 16.6 ... 17.7; bias flat at -0.02 by prediction decile.'
status: robust
metrics:
- CV
date: '2026-11-08'
source: docs/BRIEF.md 'Best pipeline on CV'
literature:
- target: '[[delange2022mindgap]]'
  relation: explains
  note: predictions shrink toward the mean when the input-target relation is imperfect
- target: '[[yang2021imbalanced]]'
  relation: suggests
  note: label-distribution smoothing for tail error
---

# CV: predictions span 66 % of the label spread; the two outer deciles hold a third of the error

Best pipeline on CV: r 0.686, residual log SD 0.138; error by label decile 16.6 ... 17.7; bias flat at -0.02 by prediction decile.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
