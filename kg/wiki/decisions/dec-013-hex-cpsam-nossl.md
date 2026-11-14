---
id: dec-013-hex-cpsam-nossl
type: decision
name: 'HEX lane: CPSAM 512 without SSL, recipe multi-target loss, five folds'
summary: 'Coordinator decision (a), 2026-11-08: train CPSAM 512 without SSL on five folds with the recipe''s multi-target loss (HEX-weighted loss dropped); stop if HEX does not go down after fold 0.'
status: accepted
date: '2026-11-08'
deciders:
- coordinator
source: experiments/lane_hex_20261108/REPORT.md 'HEX-weighted CPSAM 512'
decides:
- '[[exp-hex-cpsam512-nossl]]'
alternative:
- '[[exp-hex-hexweighted-cpsam-loss]]'
---

# HEX lane: CPSAM 512 without SSL, recipe multi-target loss, five folds

Coordinator decision (a), 2026-11-08: train CPSAM 512 without SSL on five folds with the recipe's multi-target loss (HEX-weighted loss dropped); stop if HEX does not go down after fold 0.
