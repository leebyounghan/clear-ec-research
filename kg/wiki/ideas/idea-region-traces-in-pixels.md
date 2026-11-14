---
id: idea-region-traces-in-pixels
type: idea
name: 'Traces of the marked region in pixels: compression grid, noise (plan A2)'
summary: The technician's region may leave traces (8-px JPEG grid, noise) that locate it.
status: open
origin: claude
aims_at:
- all
owner_lane: cv
cost: ~2 CPU-h
cost_hours: 2
cost_unit: CPU-h
potential_metric: CV
potential_gain: 0.003
potential_basis: 'estimate: CV reads best from three legible windows already (IoU with the true box ~0.6); for HEX the region is not the limit'
source: docs/PLAN_20261106.md row A2
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-technician-region]]'
builds_on:
- '[[method-legible-window-crop]]'
---
# Traces of the marked region in pixels: compression grid, noise (plan A2)

## Hypothesis

The technician's region may leave traces (8-px JPEG grid, noise) that locate it.

## Notes

JPEG-like traces exist in all frames (spectral bumps at 0.25 and 0.5 cycles/px, 8-px grid; lane GPU 0 report).

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
