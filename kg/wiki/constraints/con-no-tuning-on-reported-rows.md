---
id: con-no-tuning-on-reported-rows
type: constraint
name: Never choose a setting on the rows you report
summary: 'The one guard left: settings (weights, alphas, shares) are chosen on other folds (nested or cross-fit), never on the rows reported.'
status: active
origin: user
date: '2026-11-08'
source: docs/LANES_20261108.md 'Reading results'; memory/clear-ec-no-strict-statistical-gates.md
constrains:
- '[[method-nested-blend]]'
---

# Never choose a setting on the rows you report

## Rule

The one guard left: settings (weights, alphas, shares) are chosen on other folds (nested or cross-fit), never on the rows reported.
