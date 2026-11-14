---
id: idea-leave-one-source-out-harness
type: idea
name: Leave-one-source-out harness to measure source shift
summary: Hold out one label source to measure what a shift between sources costs each metric and what a correction buys.
status: closed
origin: claude
aims_at:
- all
owner_lane: backbone
source: shared notes ledger_20261111.jsonl F-SHIFT-001
novelty_check: 'KG ingest 2026-11-11: recorded after the fact from the shared ledger and lane notes; no separate KG or web search was run for this node.'
motivated_by:
- '[[find-two-sources-id-formats]]'
- '[[find-hex-test-gap]]'
tested_by:
- target: '[[exp-bb-leave-one-source-out]]'
  closeness: same
  note: penalties CD +3.4/+3.7, CV +6.0/+6.5, HEX +3.5/+0.8; 85-90 % a level offset
- target: '[[exp-bb-shading-correction]]'
  closeness: close
  note: shading correction raised the offset
uses_method:
- '[[method-leave-one-source-out]]'
---

# Leave-one-source-out harness to measure source shift

## Hypothesis

Hold out one label source to measure what a shift between sources costs each metric and what a correction buys.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Answered: [[find-source-shift-level-offset]]. Test-time alignment was not started.
