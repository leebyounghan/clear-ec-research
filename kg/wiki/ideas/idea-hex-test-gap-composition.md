---
id: idea-hex-test-gap-composition
type: idea
name: Decompose submission 1's HEX test gap before correcting for it
summary: Check whether the +0.395 HEX test-minus-OOF gap exceeds what 1000-image sampling explains before any HEX-specific adjustment.
status: closed
origin: claude
aims_at:
- HEX
owner_lane: cv
source: shared notes ledger_20261111.jsonl F-TESTGAP-001
novelty_check: 'KG ingest 2026-11-11: recorded after the fact from the shared ledger and lane notes; no separate KG or web search was run for this node.'
motivated_by:
- '[[find-hex-test-gap]]'
- '[[find-oof-test-trust]]'
tested_by:
- target: '[[exp-cv-hex-test-gap-composition]]'
  closeness: same
  note: 'inside sampling noise: joint tail 0.40, HEX alone 6.4 %'
---

# Decompose submission 1's HEX test gap before correcting for it

## Hypothesis

Check whether the +0.395 HEX test-minus-OOF gap exceeds what 1000-image sampling explains before any HEX-specific adjustment.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Answered: [[find-test-gap-within-sampling-noise]]; no HEX adjustment.
