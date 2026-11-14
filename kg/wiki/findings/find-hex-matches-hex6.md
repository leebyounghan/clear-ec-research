---
id: find-hex-matches-hex6
type: finding
name: HEX matches the six-neighbour share of markers within 1-2 % on clean sets
summary: hex6 of expert markers reproduces the HEX label within 1-2 % on clean sets.
status: robust
metrics:
- HEX
date: '2026-11-08'
source: 'docs/BRIEF.md ''Measured facts: Labels''; docs/LANES_20261108.md ''Facts per lane: HEX'''
literature:
- target: '[[viguerasguillen2022feedbackattn]]'
  relation: explains
  note: HEX by vertex vs neighbour method differ (MAE 3.14 vs 4.13); HEX underestimated below 100 cells
- target: '[[clearec2026challenge]]'
  relation: contradicts
  note: the organisers' baseline HEX is hexagon IoU, not the six-sided share our labels match
---
# HEX matches the six-neighbour share of markers within 1-2 % on clean sets

hex6 of expert markers reproduces the HEX label within 1-2 % on clean sets.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
