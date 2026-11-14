---
id: idea-from-scratch-combination-search
type: idea
name: From-scratch combination search over all members
summary: Search member combinations from scratch (forward selection, ensemble) instead of extending the incumbent stack.
status: closed
origin: claude
aims_at:
- CD
- CV
- HEX
owner_lane: cd
source: shared notes ledger_20261111.jsonl O-CD-001
novelty_check: 'KG ingest 2026-11-11: recorded after the fact from the shared ledger and lane notes; no separate KG or web search was run for this node.'
motivated_by:
- '[[find-cd-dinov3-like-members-saturated]]'
tested_by:
- target: '[[exp-cd-from-scratch-search]]'
  closeness: same
  note: '22 members, CD: forward 6.1733 (-0.0008, folds mixed), ensemble 6.1767 (+0.0026) vs 6.1741'
- target: '[[exp-cv-from-scratch-search]]'
  closeness: same
  note: 'CV, strict: grid 10.1176, forward selection 10.1324 vs fixed six 10.1160'
- target: '[[exp-hex-from-scratch-forward-selection]]'
  closeness: same
  note: 'HEX: forward selection 9.74827 vs chain 9.73258 / ridge 9.72621'
---

# From-scratch combination search over all members

## Hypothesis

Search member combinations from scratch (forward selection, ensemble) instead of extending the incumbent stack.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Closed for CD: does not beat the candidate.

## Update 2026-11-13

Closed in all three lanes: no from-scratch search beats the frozen stacks (consolidated summary, section 3.2).
