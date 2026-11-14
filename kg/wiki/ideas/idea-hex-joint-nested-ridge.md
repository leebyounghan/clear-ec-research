---
id: idea-hex-joint-nested-ridge
type: idea
name: Joint nested ridge for HEX in place of the step-by-step chain
summary: Fit all HEX stack columns in one nested ridge so no step order or step selection is chosen on reported rows.
status: adopted
origin: claude
aims_at:
- HEX
owner_lane: hex
source: shared notes ledger_20261111.jsonl D-HEX-004, O-HEX-001
novelty_check: 'KG ingest 2026-11-11: recorded after the fact from the shared ledger and lane notes; no separate KG or web search was run for this node.'
motivated_by:
- '[[con-no-tuning-on-reported-rows]]'
- '[[exp-hex-main-chain]]'
tested_by:
- target: '[[exp-hex-joint-nested-ridge]]'
  closeness: same
  note: strict 9.71977 (39 columns) vs chain procedure strict 9.74961, same run
- target: '[[exp-hex-strict-nesting-audit]]'
  closeness: close
  note: all HEX candidates strict in one run
uses_method:
- '[[method-joint-nested-ridge]]'
---

# Joint nested ridge for HEX in place of the step-by-step chain

## Hypothesis

Fit all HEX stack columns in one nested ridge so no step order or step selection is chosen on reported rows.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Adopted as the HEX freeze candidate (D-HEX-004).
