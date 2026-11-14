---
id: idea-within-donor-training
type: idea
name: Training on within-donor differences (plan G5)
summary: Fellow-eye pairs teach the model what differs between eyes.
status: closed
origin: claude
aims_at:
- CV
- HEX
source: docs/PLAN_20261106.md row G5
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-fellow-eye-correlation]]'
builds_on:
- '[[method-within-donor-training]]'
tested_by:
- target: '[[exp-gpu2-within-donor-training]]'
  closeness: same
  note: fold 0 only
---

# Training on within-donor differences (plan G5)

## Hypothesis

Fellow-eye pairs teach the model what differs between eyes.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
