---
id: idea-mape-optimal-decoding
type: idea
name: Decode each metric at the MAPE optimum (1/y-weighted median)
summary: Under MAPE the optimal point is the median of f(y)/y, below the median; train or decode with 1/y-weighted L1.
status: partially-tested
origin: literature
aims_at:
- CD
- CV
- HEX
owner_lane: cv
cost: carried by idea-distribution-head-cv-hex
cost_hours: 0
cost_unit: GPU-h
potential_metric: CV
potential_gain: null
potential_basis: remaining untested part (a distribution decoded at the median of order -1) is idea-distribution-head-cv-hex; counted there
source: paper notes demyttenaere2016mape, gneiting2011point, tofallis2015logratio
novelty_check: 'KG check 2026-11-09: close tries are the HEX MAPE-loss stackers (closed), the CD uncertainty shift (closed) and plan I4 calibration (+0.001 to +0.004); the global log-affine stretch already sets a per-metric scale. A distribution head decoded at med^-1 has not been tried.'
motivated_by:
- '[[find-cv-error-in-tails]]'
builds_on:
- '[[method-log-affine-stretch]]'
tested_by:
- target: '[[exp-hex-mape-stackers]]'
  closeness: close
  note: MAPE-loss GBM and linear fits, closed
- target: '[[exp-cd-uncertainty-shift]]'
  closeness: close
  note: per-frame shift by disagreement, closed
- target: '[[exp-hist-disagreement-calibration]]'
  closeness: close
- target: '[[exp-gpu2-frozen-pooling-heads]]'
  closeness: related
  note: quantile and HL-Gauss heads, frozen
- target: '[[exp-cd-cheap-posthoc-levers]]'
  closeness: close
  note: 'CD: second-level multipliers on top of the stretch are 1.000 +- 0.005; global multiplier +0.0046'
- target: '[[exp-cd-sub1-calibration-read]]'
  closeness: close
  note: 'CD: the candidate''s last stage is already a MAPE-optimal log-affine stretch; out-of-fold recalibration +0.0041'
literature:
- target: '[[demyttenaere2016mape]]'
  relation: suggests
- target: '[[gneiting2011point]]'
  relation: suggests
- target: '[[tofallis2015logratio]]'
  relation: suggests
- target: '[[farebrother2024hlgauss]]'
  relation: suggests
  note: histogram (HL-Gauss) head gives a predictive distribution to decode
- target: '[[kendall2017uncertainties]]'
  relation: suggests
  note: per-image predicted spread sets the shift
---
# Decode each metric at the MAPE optimum (1/y-weighted median)

## Hypothesis

Under MAPE the optimal point is the median of f(y)/y, below the median; train or decode with 1/y-weighted L1.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Update 2026-11-13

For CD the question is answered at the stack level: the last stage is a MAPE-optimal log-affine stretch and no further multiplier helps ([[find-cd-no-cheap-posthoc-lever]]). Decoding inside a member (a 1/y-weighted loss or head) for CV and HEX is still untested.
