---
id: idea-seed-audit
type: idea
name: Seed audit of every single-seed gain in the stacks
summary: Re-read every stack gain that rests on one trained seed with the seed mean before the freeze.
status: adopted
origin: claude
aims_at:
- all
owner_lane: coordinator
source: shared notes ledger_20261111.jsonl L3, O-CD-001, O-HEX-004
novelty_check: 'KG ingest 2026-11-11: recorded after the fact from the shared ledger and lane notes; no separate KG or web search was run for this node.'
motivated_by:
- '[[find-paired-ci-understates-noise]]'
- '[[find-readout-noise]]'
tested_by:
- target: '[[exp-cv-order-loss-dinov3]]'
  closeness: same
  note: '-0.0460 (seed 46) -> -0.0369 (3 seeds): not adopted'
- target: '[[exp-hex-seed-audit-chain]]'
  closeness: same
  note: cvw3 +0.0027 seed 46, -0.0029 seed 47; chain rebuilt
- target: '[[exp-cd-seed-audit-dinov3]]'
  closeness: same
  note: DINOv3 seed mean +0.0016 vs seed 46 alone
- target: '[[exp-hex-mtcnn5-seed47]]'
  closeness: same
  note: mtcnn5 gain holds with seed 47
---

# Seed audit of every single-seed gain in the stacks

## Hypothesis

Re-read every stack gain that rests on one trained seed with the seed mean before the freeze.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Done; its outcome is lesson L3 ([[con-lesson-l3-seed-mean-before-adoption]]).

## Update 2026-11-13

One adopted single-seed gain was not audited: the attention-pool column of the HEX ridge rests on seed 46 only ([[idea-attention-pool-seed-audit]]). The synthetic recipe A read also has a seed mean on fold 0 only ([[idea-synthetic-recipe-a-seed-mean-and-full-data]]).
