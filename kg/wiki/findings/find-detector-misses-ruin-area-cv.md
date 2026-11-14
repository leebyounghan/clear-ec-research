---
id: find-detector-misses-ruin-area-cv
type: finding
name: Detector misses ruin area CV; neighbour-distance CV degrades less
summary: 'With errors added to expert markers (25 sets): miss 11.5 % drops area CV r 0.89 -> 0.74 (MAPE 13.1), miss 20 % -> 0.62 (15.2); neighbour-distance CV holds 0.80 / 0.77. Jitter 10 % costs little.'
status: robust
metrics:
- CV
date: '2026-11-08'
source: 'docs/BRIEF.md ''CV from expert markers with errors added on purpose''; docs/LANES_20261108.md ''Facts per lane: CV'''
literature:
- target: '[[viguerasguillen2020dsaek]]'
  relation: corroborates
  note: CV is overestimated when few cells are detected
- target: '[[fabijanska2018unet]]'
  relation: corroborates
  note: 4.5 % count error already gives ~12 % CV error
- target: '[[viguerasguillen2019fcn]]'
  relation: corroborates
  note: 0.5-0.9 % wrongly split/merged cells cost 2.8-3.6 % relative CV error
- target: '[[piorkowski2017segmentation]]'
  relation: corroborates
  note: side-length CV stable across segmentations; HEX method-sensitive
- target: '[[shumilin2023voronoi]]'
  relation: explains
  note: differentiable Voronoi moves points but cannot add or delete them
---
# Detector misses ruin area CV; neighbour-distance CV degrades less

With errors added to expert markers (25 sets): miss 11.5 % drops area CV r 0.89 -> 0.74 (MAPE 13.1), miss 20 % -> 0.62 (15.2); neighbour-distance CV holds 0.80 / 0.77. Jitter 10 % costs little.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
