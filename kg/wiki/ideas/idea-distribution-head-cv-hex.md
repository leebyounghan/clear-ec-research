---
id: idea-distribution-head-cv-hex
type: idea
name: Distribution head for CV and HEX with label-noise width growing with CV, decoded at the MAPE optimum
summary: Train CV (and HEX) as label distributions (DLDL / HL-Gauss) with a per-image width that grows with the CV level and is at least the sampling SE, and decode the 1/y-weighted median per image instead of the expectation.
status: open
origin: literature
aims_at:
- CV
- HEX
owner_lane: cv
cost: ~4 GPU-h (DINOv3-B, folds 0,4) to ~10 GPU-h (five folds); a frozen-feature version ~1 GPU-h
cost_hours: 4
cost_unit: GPU-h
potential_metric: CV
potential_gain: 0.015
potential_basis: 'estimate: upper end of single member/head changes in the CV lane (DINOv3 seeds -0.010 to -0.0115 alone, frozen heads part of -0.037); frozen HL-Gauss read +0.036 and the quantile head added +0.0002, so the fixed-width, expectation-decoded version has shown nothing yet'
source: Literature review 2026-11-09 (paper notes' suggests_try), coordinator list of eight suggestions, suggestion 3
novelty_check: 'KG check 2026-11-09 (kg_query check + grep over wiki/): check distribution head; check hl-gauss; check quantile -> exp-gpu2-frozen-pooling-heads (frozen, closed), exp-cd-dinov3-distribution-head (CD lane, queued 11-09: fixed sigma, expectation decoding). Literature: papers from the KG literature review (gao2017dldl, farebrother2024hlgauss, doughty2020cvcells, doughty2018regional, pan2018meanvariance, stewart2023regclass); no new web search in this pass.'
motivated_by:
- '[[find-cv-label-discrete]]'
- '[[find-sampling-se-150-cells]]'
- '[[find-lit-cv-hex-grader-noise]]'
- '[[idea-mape-optimal-decoding]]'
builds_on:
- '[[method-dinov3-b16]]'
- '[[method-frozen-token-heads]]'
tested_by:
- target: '[[exp-cd-dinov3-distribution-head]]'
  closeness: close
  note: 'closed in the CD lane: HL-Gauss end to end on DINOv3-B, sigma 0.75 bin fixed, expectation decoding, +0.008 on five folds for CD; no CV / HEX read of it is recorded'
- target: '[[exp-gpu2-frozen-pooling-heads]]'
  closeness: close
  note: 'frozen recipe A heads: HL-Gauss +0.036, quantile head -0.029 alone but added +0.0002; fixed width, not decoded at the MAPE optimum'
literature:
- target: '[[gao2017dldl]]'
  relation: source-of-method
  note: Gaussian label distributions, KL loss
- target: '[[farebrother2024hlgauss]]'
  relation: source-of-method
  note: HL-Gauss histogram head
- target: '[[doughty2020cvcells]]'
  relation: suggests
  note: 'CV label noise grows with CV: width proportional to label'
- target: '[[doughty2018regional]]'
  relation: suggests
  note: heteroscedastic regional sampling noise
- target: '[[gneiting2011point]]'
  relation: suggests
  note: decode the median of order -1
- target: '[[kendall2017uncertainties]]'
  relation: suggests
  note: per-image predicted width
- target: '[[pan2018meanvariance]]'
  relation: explains
  note: 'no variance penalty: the spread is needed for decoding'
- target: '[[stewart2023regclass]]'
  relation: suggests
  note: weak support for binned heads
---
# Distribution head for CV and HEX with label-noise width growing with CV, decoded at the MAPE optimum

## Hypothesis

CV and HEX labels are discrete and noisy with a width that grows with CV; a distribution head trained with that width and decoded at the MAPE optimum lowers CV MAPE where a point head regresses to the mean.

## Closest tries in the KG

- [[exp-gpu2-frozen-pooling-heads]] (close, closed): frozen HL-Gauss +0.036; quantile head added +0.0002.
- [[exp-cd-dinov3-distribution-head]] (close, running): end-to-end HL-Gauss on DINOv3-B, fixed width, expectation decoding, CD read on folds 0,4.

## Literature

gao2017dldl, farebrother2024hlgauss, doughty2020cvcells, doughty2018regional, gneiting2011point, kendall2017uncertainties, pan2018meanvariance, stewart2023regclass.

## Minimal test

Cheapest: read CV and HEX from the CD lane's queued HL-Gauss DINOv3-B run (folds 0,4) decoded two ways (expectation vs 1/y-weighted median), then add the member to the CV lane best (10.114) on folds 0,4 against the same member with the recipe head. If it lowers CV, run five folds with a CV-dependent width.

## Risks

Two-fold reads overstate ([[find-screen-folds-overstate]]); decoding gains may already be absorbed by the global stretch.

## Update 2026-11-13

The CD lane's distribution-head member is closed at +0.008 on five folds (CD). The CV / HEX form with a label-noise width and MAPE-optimal decoding was not run; status stays open.
