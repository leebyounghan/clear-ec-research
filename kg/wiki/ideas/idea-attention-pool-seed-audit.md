---
id: idea-attention-pool-seed-audit
type: idea
name: Seed audit of the attention-pool column in the HEX ridge
summary: The adopted attention-pool DINOv3-B column of the HEX joint ridge rests on seed 46 only; the seed-mean rule (L3) was not applied to it.
status: never-tried
origin: claude
aims_at:
- HEX
owner_lane: hex
cost: not estimated in the sources (five fold runs and one full-data build of a second seed)
potential_metric: HEX
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: docs/lane_summaries_20261113/final_summary_hex_20261113.md sections 4c and 6; docs/lane_summaries_20261113/final_summary_backbone_20261113.md section 1; docs/FINAL_SUMMARY_20261113.md sections 3.3 and 5
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[con-lesson-l3-seed-mean-before-adoption]]'
- '[[exp-hex-joint-nested-ridge]]'
tested_by:
- target: '[[exp-hex-joint-nested-ridge]]'
  closeness: related
  note: 'the column itself, seed 46: strict 9.72662 -> 9.71977'
- target: '[[exp-hex-deploy-tiers]]'
  closeness: related
  note: removing the column from t2 costs +0.0080 (9.72093 -> 9.72891), still seed 46
- target: '[[exp-hex-mtcnn5-seed47]]'
  closeness: related
  note: 'the same audit done for the CPSAM multi-target column: the second seed keeps the gain'
depends_on:
- '[[idea-seed-audit]]'
---

# Seed audit of the attention-pool column in the HEX ridge

The adopted attention-pool DINOv3-B column of the HEX joint ridge rests on seed 46 only; the seed-mean rule (L3) was not applied to it.

## Hypothesis

The column is worth -0.0069 HEX strict (39 vs 38 columns) and +0.0080 when removed from tier t2, both above the 0.005 seed-noise size, but from one trained seed. A second seed would show how much of it is seed luck.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Risks

Tiers p, t1, t2 and full load the column; t3 does not.

## Outcome 2026-11-13

Open at close-out; listed as unverified in the consolidated summary (section 5).
