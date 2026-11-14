---
id: idea-drop-path
type: idea
name: Drop path under recipe A
summary: Stochastic depth regularises the encoder.
status: closed
origin: claude
aims_at:
- all
source: 'docs/LANES_20261108.md ''Facts per lane: Backbone'''
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[exp-hist-encoder-lr-x5]]'
builds_on:
- '[[method-drop-path-regularisation]]'
tested_by:
- target: '[[exp-bb-droppath-ema-swa]]'
  closeness: same
  note: fold 0 neutral (+0.027; added -0.002)
---

# Drop path under recipe A

## Hypothesis

Stochastic depth regularises the encoder.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
