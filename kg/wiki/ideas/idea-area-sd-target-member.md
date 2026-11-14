---
id: idea-area-sd-target-member
type: idea
name: A member trained on the area-SD target for CV
summary: Train a member on the SD of cell area (CV = SD / mean area) so the stack gets the numerator of CV directly.
status: closed
origin: claude
aims_at:
- CV
owner_lane: cv
source: cv_lane_notes_20261108.md '2026-11-11 03:15 EST'; shared notes ledger_20261111.jsonl L1
novelty_check: 'KG ingest 2026-11-11: recorded after the fact from the shared ledger and lane notes; no separate KG or web search was run for this node.'
motivated_by:
- '[[find-cv-error-in-tails]]'
tested_by:
- target: '[[exp-cv-area-sd-member]]'
  closeness: same
  note: zero or worse on top of the 6-component CV stack; number not recorded
---

# A member trained on the area-SD target for CV

## Hypothesis

Train a member on the SD of cell area (CV = SD / mean area) so the stack gets the numerator of CV directly.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Closed: zero or worse on the CV stack; the value is not recorded.

## Update 2026-11-13

Number now on record: stack -0.0358 with the member against -0.0411 without ([[exp-cv-area-sd-member]]).
