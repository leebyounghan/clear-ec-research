---
id: idea-residual-diversity-gate
type: idea
name: Residual-diversity gate before training a new member (L1)
summary: Spend GPU time on a new member only if its residual correlation with recipe A is below 0.9 on every metric; nothing the backbone lane trained met that bar.
status: partially-tested
origin: claude
aims_at:
- all
owner_lane: backbone
cost: CPU only (a residual correlation on stored predictions)
potential_metric: all
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: docs/lane_summaries_20261113/final_summary_backbone_20261113.md section 4c (idea-residual-diversity-gate); shared notes ledger_20261111.jsonl L1 (line 2)
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[con-lesson-l1-member-alone-vs-stack]]'
- '[[find-better-alone-not-stack-weight]]'
- '[[find-other-families-err-differently]]'
tested_by:
- target: '[[exp-bb-synth-recipe-a]]'
  closeness: related
  note: 'synthetic recipe A: residual r with plain recipe A 0.985-0.996 per metric on every fold; no stack gain'
- target: '[[exp-bb-recipe-a-window-view]]'
  closeness: related
  note: 'recipe A on the 2x window: 0.93 / 0.97 / 0.98; CD stack +0.0026'
- target: '[[exp-bb-cpsam512-two-fold]]'
  closeness: related
  note: 'CPSAM 512: 0.89 / 0.93 / 0.95'
- target: '[[exp-cd-cdonly-recipe-a-elr3e5]]'
  closeness: related
  note: CD-only training raised the residual correlation with the other members (0.934 vs 0.914 with P1) and the stack got worse
---

# Residual-diversity gate before training a new member (L1)

Spend GPU time on a new member only if its residual correlation with recipe A is below 0.9 on every metric; nothing the backbone lane trained met that bar.

## Hypothesis

A member adds to a stack through errors that differ from the incumbent's, not through its score alone.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-13

Nothing below 0.9 on every metric was found among the backbone lane's members.
