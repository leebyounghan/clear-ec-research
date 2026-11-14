---
id: idea-t4-precision-hex
type: idea
name: Reduce the HEX deviation of the T4 path
summary: On the T4 path the HEX output deviates from the validated one by 0.44-0.45 % median (max 3.9 %), an estimated HEX impact of +0.018; fp32 does not reduce it.
status: partially-tested
origin: claude
aims_at:
- HEX
owner_lane: deploy
cost: not estimated in the sources
potential_metric: HEX
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: docs/lane_summaries_20261113/final_summary_hex_20261113.md section 4c (idea-t4-precision-hex); shared notes ledger_20261111.jsonl O-DEPLOY-020 (line 185), O-DEPLOY-021 (line 188), O-COORD-F32-200 (line 203)
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[find-kernel-change-not-precision-moves-outputs]]'
- '[[exp-deploy-t4-default-200-rows-p]]'
tested_by:
- target: '[[exp-deploy-t4-all-fp32-200-rows]]'
  closeness: close
  note: 'all-fp32: HEX impact +0.0187 against fp16''s +0.0178, no closer to the reference'
- target: '[[exp-deploy-t4-precision-paths-20-frames]]'
  closeness: close
  note: mix (detectors / hex6 in fp32) HEX median 0.437 %, all-fp32 0.436 %, A10G proxy 0.426 %
- target: '[[exp-deploy-colab-t4-timing-vm2]]'
  closeness: related
  note: fp16 with only hex6_correction in fp32 takes 364.9 s on a real T4, over the limit
---

# Reduce the HEX deviation of the T4 path

On the T4 path the HEX output deviates from the validated one by 0.44-0.45 % median (max 3.9 %), an estimated HEX impact of +0.018; fp32 does not reduce it.

## Hypothesis

HEX carries about 3/4 of the T4 path's score impact because thresholded detector peaks flip under a kernel change. A detector path whose peaks are stable across kernels would remove most of it.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-13

Dtype changes are closed (no gain, and too slow on a T4). A kernel-stable detector path was not tried.
