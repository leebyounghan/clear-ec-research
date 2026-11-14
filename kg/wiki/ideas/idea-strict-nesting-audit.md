---
id: idea-strict-nesting-audit
type: idea
name: Strict-nesting audit of every stack number
summary: Recompute every stack, chain and selection number with the scored fold's labels hidden in the whole procedure.
status: adopted
origin: claude
aims_at:
- all
owner_lane: coordinator
source: shared notes ledger_20261111.jsonl L4, L5, O-HEX-001, O-VER-001
novelty_check: 'KG ingest 2026-11-11: recorded after the fact from the shared ledger and lane notes; no separate KG or web search was run for this node.'
motivated_by:
- '[[con-no-tuning-on-reported-rows]]'
tested_by:
- target: '[[exp-hex-strict-nesting-audit]]'
  closeness: same
  note: chain 9.73258 -> 9.74961 strict
- target: '[[exp-cv-strict-nesting-audit]]'
  closeness: same
  note: 10.11014 -> 10.11596
- target: '[[exp-hex-cellroute-konan-step]]'
  closeness: same
  note: -0.01301 as computed -> -0.00306 strict on the same base
- target: '[[exp-ref-freeze-candidate-integrated]]'
  closeness: same
  note: verifier table
- target: '[[exp-cd-deploy-tiers]]'
  closeness: same
  note: 'CD: the recipe A snapshot and ViT-L single views add nothing under strict nesting (t1 6.1737 vs full 6.1743)'
uses_method:
- '[[method-strict-nesting]]'
---

# Strict-nesting audit of every stack number

## Hypothesis

Recompute every stack, chain and selection number with the scored fold's labels hidden in the whole procedure.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Done; its outcome is lessons L4 and L5.

## Update 2026-11-13

Sizes of the leak it removed (consolidated summary): 0.006 (CV inner stretch), 0.010 (HEX chain), 0.010 (cell-route version choice). CD: v3 +0.0002, best pipeline -0.0028 (11-11); and the full candidate's extra CD weights buy nothing under strict nesting ([[find-cd-full-extra-weights-nothing-over-t1]]).
