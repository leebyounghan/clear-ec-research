---
id: idea-hidden-acquisition-batches
type: idea
name: Correct hidden acquisition batches (plan A1)
summary: The -22 rows are worst on all three metrics; find and correct the batch from the image.
status: closed
origin: claude
aims_at:
- all
source: docs/PLAN_20261106.md row A1
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-minus22-processing-change]]'
builds_on:
- '[[method-per-source-correction]]'
tested_by:
- target: '[[exp-gpu0-minus22-correction]]'
  closeness: same
---

# Correct hidden acquisition batches (plan A1)

## Hypothesis

The -22 rows are worst on all three metrics; find and correct the batch from the image.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
