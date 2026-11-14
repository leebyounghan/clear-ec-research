---
id: idea-mape-shift-predicted-noise
type: idea
name: Per-image MAPE-optimal shift from a predicted noise scale (1/y-weighted median)
summary: 'Shift each prediction toward the median of order -1 (median x exp(-s_i^2) for log-normal error) using a per-image noise scale s_i from a heteroscedastic head. Closed: the same mechanism (per-image shift by an uncertainty estimate, fitted on OOF MAPE) is closed as tested on CD (three proxies) and HEX (exact-MAPE per-image GBM and MAPE-loss linear fit).'
status: closed
origin: literature
aims_at:
- CD
- CV
- HEX
owner_lane: cv
cost: ~1 CPU-h on stored predictions; ~4 GPU-h if a log-variance head is trained
cost_hours: 1
cost_unit: CPU-h
potential_metric: CV
potential_gain: null
potential_basis: 'closed: CD shift k chosen near 0 (+0.012 to +0.025 worse), HEX exact-MAPE GBM +0.055 / +0.031, MAPE-loss linear fit +0.0007; the one untested input (a trained aleatoric head) is carried by idea-distribution-head-cv-hex'
source: Literature review 2026-11-09 (paper notes' suggests_try), coordinator list of eight suggestions, suggestion 2
novelty_check: 'KG check 2026-11-09 (kg_query check + grep over wiki/): check mape; check uncertainty shift -> exp-cd-uncertainty-shift (closed, third version), exp-hex-mape-stackers (closed), exp-hist-disagreement-calibration (closed, CV lane list), idea-mape-optimal-decoding (partially tested). Literature: papers from the KG literature review (demyttenaere2016mape, gneiting2011point, kendall2017uncertainties, tofallis2015logratio); no new web search in this pass.'
motivated_by:
- '[[idea-mape-optimal-decoding]]'
- '[[find-cv-error-in-tails]]'
builds_on:
- '[[method-uncertainty-mape-shift]]'
- '[[method-log-affine-stretch]]'
tested_by:
- target: '[[exp-cd-uncertainty-shift]]'
  closeness: same
  note: 'same mechanism on CD: one-parameter per-image shift by member disagreement (third version after perturbation sensitivity and legibility bands), nested, +0.012 to +0.025, lane stop. Differs: the noise proxy is disagreement, not a trained aleatoric head'
- target: '[[exp-hex-mape-stackers]]'
  closeness: same
  note: 'same mechanism on HEX: GBM on y/HB weighted HB/y learns the exact-MAPE per-image ratio from features (+0.055 / +0.031 with green features); MAPE-loss linear fit on uncertainty features +0.0007. ''Per-image MAPE shift does not beat the global stretch'''
- target: '[[exp-hist-disagreement-calibration]]'
  closeness: close
  note: plan I4 frame-level MAPE calibration from disagreement proxies, +0.001 to +0.004 (folds not stated); the CV lane's closed list
literature:
- target: '[[demyttenaere2016mape]]'
  relation: source-of-method
  note: MAPE = 1/y-weighted MAE
- target: '[[gneiting2011point]]'
  relation: source-of-method
  note: APE elicits the median of order -1
- target: '[[kendall2017uncertainties]]'
  relation: source-of-method
  note: learned per-image log-variance (Laplace form)
- target: '[[tofallis2015logratio]]'
  relation: explains
  note: MAPE rewards slight under-prediction
---
# Per-image MAPE-optimal shift from a predicted noise scale (1/y-weighted median)

## Hypothesis

Under MAPE the optimum lies below the median by an amount that grows with the image's predictive spread; a per-image shift from a predicted noise scale should beat the global stretch.

## Closest tries in the KG

- [[exp-cd-uncertainty-shift]] (same mechanism, CD, closed after three versions).
- [[exp-hex-mape-stackers]] (same mechanism, HEX, closed: exact-MAPE GBM worse, linear fit +0.0007).
- [[exp-hist-disagreement-calibration]] (close, CV lane closed list).

Verdict 2026-11-09: these are the same test as far as the mechanism goes; what differs is only the noise estimate (a trained aleatoric head instead of disagreement or uncertainty features). The global part of the shift is already in the log-affine stretch. Not worth rerunning as a shift; the remaining variant (a predicted distribution decoded at its median of order -1) is [[idea-distribution-head-cv-hex]].

## Literature

demyttenaere2016mape, gneiting2011point, kendall2017uncertainties, tofallis2015logratio.

## Minimal test

None (closed). If idea-distribution-head-cv-hex is run, its per-image width gives the shift for free.

## Risks

Per-image shifts add variance; three proxies already found nothing on CD.
