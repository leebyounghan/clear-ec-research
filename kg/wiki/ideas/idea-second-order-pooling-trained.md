---
id: idea-second-order-pooling-trained
type: idea
name: Second-order token pooling trained end to end
summary: Covariance or spectrum pooling trained with the encoder, not frozen.
status: partially-tested
origin: claude
aims_at:
- CV
- CD
owner_lane: backbone
cost: ~12 GPU-h
cost_hours: 12
cost_unit: GPU-h
potential_metric: CV
potential_gain: 0.005
potential_basis: 'estimate: frozen spectrum/cov_sqrt/autocorr heads are already in the CV stack (part of -0.037); end-to-end training would add only what the frozen heads miss'
source: docs/ASSUMPTION_AUDIT_20261106.md belief 7; docs/PLAN_20261106.md row G3
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-std-pooling-beats-mean]]'
builds_on:
- '[[method-frozen-token-heads]]'
tested_by:
- target: '[[exp-gpu2-frozen-pooling-heads]]'
  closeness: close
  note: frozen encoder only
- target: '[[exp-cv-frozen-token-heads]]'
  closeness: close
  note: frozen heads kept as a CV correction (-0.012)
---
# Second-order token pooling trained end to end

## Hypothesis

Covariance or spectrum pooling trained with the encoder, not frozen.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
