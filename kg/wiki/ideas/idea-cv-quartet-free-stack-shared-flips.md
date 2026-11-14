---
id: idea-cv-quartet-free-stack-shared-flips
type: idea
name: A CV stack without the graph quartets that shares detector flips with HEX
summary: Dropping both graph quartets from the CV stack costs +0.0118 strict and saves 120 graph-net files and, only if no HEX part needs the flipped pre-only peaks, up to 3 of the 4 flips of the pre-only detector pass.
status: open
origin: claude
aims_at:
- CV
owner_lane: cv
cost: CPU check of the HEX inputs; not estimated in the sources
potential_metric: CV
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: docs/lane_summaries_20261113/final_summary_cv_20261113.md section 4c; docs/FINAL_SUMMARY_20261113.md section 3.3; shared notes ledger_20261111.jsonl O-CV-TIERS-001 (line 150)
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[exp-cv-deploy-tiers]]'
- '[[con-platform-runtime-300s-t4]]'
tested_by:
- target: '[[exp-cv-deploy-tiers]]'
  closeness: close
  note: t1 without quartets 10.1233 against t1 10.1115 (+0.0118 strict); the saving on the HEX side was not measured
- target: '[[exp-deploy-flops-per-frame]]'
  closeness: related
  note: pre-only detectors 225 TFLOP per frame
---

# A CV stack without the graph quartets that shares detector flips with HEX

Dropping both graph quartets from the CV stack costs +0.0118 strict and saves 120 graph-net files and, only if no HEX part needs the flipped pre-only peaks, up to 3 of the 4 flips of the pre-only detector pass.

## Hypothesis

The pre-only detectors are 225 of about 1900 TFLOP per frame; if HEX does not read the flipped peaks, a quartet-free CV stack would cut that pass. Whether HEX needs the flipped pre-only peaks was not checked.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-13

Open at close-out: a runtime lever with a known CV price and an unknown saving.
