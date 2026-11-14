---
id: con-12h-full-data-retrain
type: constraint
name: Adopted models must retrain on full data in 12 h on one GPU
summary: An adopted model must retrain on full data in 12 h or less on one GPU (rebuild day 11-12).
status: active
origin: claude
date: '2026-11-07'
source: docs/LANES_20261108.md 'Rules that do not change'
constrains:
- '[[exp-bb-cpsam512-two-fold]]'
---

# Adopted models must retrain on full data in 12 h on one GPU

## Rule

An adopted model must retrain on full data in 12 h or less on one GPU (rebuild day 11-12).

## Notes

CPSAM 512 with SSL (8.6 h + 8.6 h) exceeds it; ViT-L full data ~4.4 h fits (CD lane timing).
