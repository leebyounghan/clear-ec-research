---
id: idea-strong-view-ssl
type: idea
name: SSL with strong, domain-specific views (plan G2)
summary: Crop and broad optics views make SSL features stronger.
status: closed
origin: claude
aims_at:
- all
source: docs/PLAN_20261106.md row G2
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[exp-hist-ssl-duration]]'
builds_on:
- '[[method-strong-view-ssl]]'
tested_by:
- target: '[[exp-gpu1-strong-view-ssl-e2e]]'
  closeness: same
- target: '[[exp-gpu1-strong-view-ssl-frozen]]'
  closeness: same
---

# SSL with strong, domain-specific views (plan G2)

## Hypothesis

Crop and broad optics views make SSL features stronger.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
