---
id: find-technician-region
type: finding
name: The technician marks about 150 cells in a fixed 538 x 408 box (~17 % of the frame)
summary: Median 148 markers (p10-p90 119-158); hull area correlates -0.93 with CD; the window with most confident peaks lies mostly inside the box.
status: robust
metrics:
- CD
- CV
- HEX
date: '2026-11-08'
source: docs/BRIEF.md 'The region a technician marks'
literature:
- target: '[[benetz2024donorecd]]'
  relation: corroborates
  note: eye-bank technician-counted region; matching it halves-to-thirds the CD bias
- target: '[[joseph2020postkeratoplasty]]'
  relation: corroborates
  note: readers annotate a sub-region; 564 acceptable cells outside it
- target: '[[mccarey2008review]]'
  relation: corroborates
  note: 'centre method: ~150 dotted cells give ~100 areas'
- target: '[[doughty2018regional]]'
  relation: explains
  note: CV depends on which and how many cells are sampled
---
# The technician marks about 150 cells in a fixed 538 x 408 box (~17 % of the frame)

Median 148 markers (p10-p90 119-158); hull area correlates -0.93 with CD; the window with most confident peaks lies mostly inside the box.

## Limits and notes

Window IoU with the box is reported as 0.42 in BRIEF and 0.57-0.61 / 0.605 elsewhere (see exp-hist-region-from-image, ambiguous).

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
