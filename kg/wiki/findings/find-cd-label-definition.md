---
id: find-cd-label-definition
type: finding
name: CD = 1e6 / integer mean cell area; label / marker density 0.97-1.43
summary: CD labels are 1e6 divided by the integer mean cell area; label against expert-marker density runs 0.97-1.43, near 1 on clean sets.
status: robust
metrics:
- CD
date: '2026-11-08'
source: 'docs/BRIEF.md ''Measured facts: Labels''; docs/LANES_20261108.md ''Facts per lane: CD'''
literature:
- target: '[[takahashi1996konanpatent]]'
  relation: corroborates
  note: CD = 1e6 / mean cell area in the Konan centre method
- target: '[[tey2025widefield]]'
  relation: corroborates
  note: 'a segmentation with its own CD formula lands far from device values: match the label formula'
- target: '[[clearec2026challenge]]'
  relation: explains
  note: organisers' baseline CD = N x 1e6 / total area; device and label procedure not published
- target: '[[kervadec2019constrained]]'
  relation: explains
  note: CD is an image-level size statistic that can supervise masks
- target: '[[huang2017autovsmanual]]'
  relation: explains
  note: device ECD bias vs manual depends on age/CV and cell size
---
# CD = 1e6 / integer mean cell area; label / marker density 0.97-1.43

CD labels are 1e6 divided by the integer mean cell area; label against expert-marker density runs 0.97-1.43, near 1 on clean sets.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
