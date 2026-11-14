---
id: idea-donor-memory
type: idea
name: Donor memory from images (plan I7)
summary: Retrieve the fellow eye and borrow its residual.
status: closed
origin: claude
aims_at:
- all
source: docs/PLAN_20261106.md row I7
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-fellow-eye-correlation]]'
builds_on:
- '[[method-donor-memory]]'
tested_by:
- target: '[[exp-gpu1-donor-memory]]'
  closeness: same
- target: '[[exp-cd-fellow-eye-residual-transfer]]'
  closeness: close
  note: 'fellow-eye residual transfer on tier p: oracle with IDs CD -0.236, image-only strict kNN +0.002 to +0.009; closed by the PI as a trick'
---

# Donor memory from images (plan I7)

## Hypothesis

Retrieve the fellow eye and borrow its residual.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Update 2026-11-13

A second form was read on 11-13 and closed by the PI as a trick, not to be used: transferring the fellow eye's residual ([[exp-cd-fellow-eye-residual-transfer]]). With IDs as an oracle it reads CD -0.236, CV -0.200, HEX -0.091 on tier p; from the image alone the partner is rarely findable (top-1 4.9 %) and the strict kNN transfer gains nothing (+0.0016 to +0.0088).
