---
id: idea-hex-only-token-corrections
type: idea
name: Token-statistic corrections applied to HEX only, chosen nested
summary: Lane GPU 2 saw most token blocks lower HEX while raising CD; HEX-only -0.0126 was chosen after seeing the outer folds.
status: open
origin: claude
aims_at:
- HEX
owner_lane: hex
cost: ~1 CPU-h (nested read on stored token statistics)
cost_hours: 1
cost_unit: CPU-h
potential_metric: HEX
potential_gain: 0.003
potential_basis: 'estimate: the exploratory -0.0126 was chosen after seeing the outer folds; the HEX chain has converged (second passes change nothing)'
source: experiments/lane_gpu2_20261106/REPORT.md 'Corrections from token statistics' (exploratory)
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[exp-gpu2-token-stat-corrections]]'
builds_on:
- '[[method-frozen-token-heads]]'
tested_by:
- target: '[[exp-gpu2-token-stat-corrections]]'
  closeness: close
  note: metric chosen after seeing outer folds, so not a valid read
- target: '[[exp-hex-gnn-variants]]'
  closeness: related
  note: image statistics in the graph head
---
# Token-statistic corrections applied to HEX only, chosen nested

## Hypothesis

Lane GPU 2 saw most token blocks lower HEX while raising CD; HEX-only -0.0126 was chosen after seeing the outer folds.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
