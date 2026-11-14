---
id: find-region-not-limit-for-hex
type: finding
name: For HEX the region is not the limit; cell-level error is
summary: Mask hex6 inside the box r 0.77 vs 0.73 whole frame; predicted boxes 0.75-0.79.
status: robust
metrics:
- HEX
date: '2026-11-08'
source: experiments/lane_hex_20261108/REPORT.md 'Segmentation'
literature:
- target: '[[viguerasguillen2022feedbackattn]]'
  relation: explains
  note: HEX needs ~100+ cells; error is cell-level
---
# For HEX the region is not the limit; cell-level error is

Mask hex6 inside the box r 0.77 vs 0.73 whole frame; predicted boxes 0.75-0.79.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
