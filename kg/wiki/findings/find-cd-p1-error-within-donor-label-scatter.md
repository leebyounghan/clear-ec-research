---
id: find-cd-p1-error-within-donor-label-scatter
type: finding
name: CD error sits in source P1 (7.04 vs 5.29 for P2), mostly through label scatter within a donor (log-label SD 0.137 vs 0.058)
summary: 'On the frozen candidate''s strict CD OOF, P1 reads 7.04 and carries 57.5 % of the error, P2 5.29. The gap is not the label range and only partly image quality; within a donor the log label scatters with SD 0.137 in P1 against 0.058 in P2. The within-donor figure is a proxy: the ids do not say which eye.'
status: tentative
metrics:
- CD
date: '2026-11-13'
source: shared notes ledger_20261111.jsonl O-CDANAT-001 (line 93), O-CDANAT-002 (line 94); experiments/lane_cd_20261108/reads/cd_error_anatomy_20261111.json; docs/lane_summaries_20261113/final_summary_cd_20261113.md section 4c
motivated_by:
- '[[find-two-sources-id-formats]]'
---

# CD error sits in source P1 (7.04 vs 5.29 for P2), mostly through label scatter within a donor (log-label SD 0.137 vs 0.058)

On the frozen candidate's strict CD OOF, P1 reads 7.04 and carries 57.5 % of the error, P2 5.29. The gap is not the label range and only partly image quality; within a donor the log label scatters with SD 0.137 in P1 against 0.058 in P2. The within-donor figure is a proxy: the ids do not say which eye.

## Evidence

[[exp-cd-error-anatomy]]: P1 at P2's label mix 8.51, P2 at P1's mix 5.10; noise floor 3.58 vs 2.53, interior cells median 347 vs 671, still 5.80 vs 4.60 on frames with >= 150 cells; within-donor residual 0.083 vs 0.043; prediction follows label moves r 0.80 / 0.71. Six impossible-label serials (2020-0644..0651) are worth 0.187 MAPE. Bounds: P1 at P2's error 5.29; 150-cell sampling floor 2.49 %.

## Limits

A proxy, not a measurement of label noise: two frames of one donor may be two eyes. One candidate, one strict OOF.

## Bears on

No action is possible without relabelling, which the PI rules out ([[con-no-human-annotation]]). Earlier source-level reads: [[find-two-sources-id-formats]], [[find-impossible-labels]].
