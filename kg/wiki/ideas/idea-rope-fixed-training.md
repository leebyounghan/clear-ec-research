---
id: idea-rope-fixed-training
type: idea
name: Train with the RoPE coordinate rescale off
summary: Whether the random rescale noise itself costs anything.
status: open
origin: claude
aims_at:
- all
owner_lane: backbone
cost: ~4 GPU-h (fold 0,4 recipe A)
cost_hours: 4
cost_unit: GPU-h
potential_metric: all
potential_gain: null
potential_basis: 'unknown: no read of training without the rescale noise; inference-time scale only was tested'
source: experiments/lane_gpu0_20261106/REPORT.md (queued as rope_fixed; no result in the sources)
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[exp-gpu0-rope-scale-inference]]'
builds_on:
- '[[method-rope-scale]]'
tested_by:
- target: '[[exp-gpu0-rope-scale-inference]]'
  closeness: related
  note: inference-time scale only
---
# Train with the RoPE coordinate rescale off

## Hypothesis

Whether the random rescale noise itself costs anything.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
