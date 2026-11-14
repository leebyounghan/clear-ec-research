---
id: find-flip-tta-single-member-small
type: finding
name: Flip TTA on a single final-epoch member is worth almost nothing (-0.002)
summary: Backbone drop-path member, fold 0.
status: tentative
metrics:
- mean
date: '2026-11-08'
source: experiments/lane_backbone_20261108/RESULTS.tsv droppath02_fold0_tta
literature:
- target: '[[wang2019tta]]'
  relation: contradicts
  note: segmentation TTA gains ~1 Dice; flip TTA on one final-epoch regression member is worth -0.002
---
# Flip TTA on a single final-epoch member is worth almost nothing (-0.002)

Backbone drop-path member, fold 0.

## Limits and notes

Apparent conflict: the CD lane reports four-flip TTA over epochs 16-20 lowering a family member's own CD by 0.05-0.1. The CD-lane version also averages five epochs and uses other encoders, so the two are not the same test.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
