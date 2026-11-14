---
id: kiet2026psc
type: paper
title: "Predictive Spectral Calibration for Source-Free Test-Time Regression"
authors: [Nguyen Viet Tuan Kiet, Huynh Thanh Trung, Pham Huy Hieu]
year: 2026
venue: "arXiv"
doi: ""
arxiv: "2603.09338"
pmcid: ""
url: "https://arxiv.org/html/2603.09338"
citation: "Kiet NVT, Trung HT, Hieu PH. Predictive Spectral Calibration for Source-Free Test-Time Regression. arXiv:2603.09338. 2026"
tier: candidate
group: t7-domain-shift
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html]
evidence: preprint
tldr: "Extends SSA regression TTA with block spectral matching over a probe bank plus a residual term in the orthogonal complement; still updates only normalisation affine parameters with stored source statistics; modest gains over SSA on SVHN->MNIST and UTKFace-C."
relevance_clearec: low
relevance_metrics: [CD, CV, HEX]
relevance_note: "Confirms regression TTA remains a thin, natural-image-only literature in 2026; gains over SSA are small, so it does not change our TTA choice."
code_release: unknown
weights_release: unknown
data_scale: "SVHN->MNIST; UTKFace with 13 corruption types"
key_numbers:
  - "SVHN->MNIST R2: source 0.239, BN-adapt 0.341, SSA 0.452, PSC 0.473 (lambda 0), oracle 0.866 (verified-from-source)"
  - "UTKFace-C mean R2: BN-adapt 0.563, SSA 0.583, PSC 0.604 (lambda 0) / 0.619 (lambda 1), oracle 0.697 (verified-from-source)"
  - "Best lambda is task-dependent (0 for SVHN->MNIST, 1 for UTKFace-C) (verified-from-source)"
results:
  - {task: "age regression under corruption", dataset: "UTKFace-C (mean)", metric: "R2", value: "0.619", comparator: "SSA", comparator_value: "0.583", locator: "Table 2", provenance: "verified-from-source"}
suggests_try: "Nothing actionable beyond SSA; if regression TTA is tried, start with SSA (simpler, published) and treat PSC as an optional variant."
tags: [test-time-adaptation, regression, source-free, spectral-alignment]
search_keywords: [PSC, predictive spectral calibration, test-time regression, SSA extension]
uses_dataset: [SVHN, MNIST, UTKFace]
uses_method: [subspace alignment, symmetric KL, normalisation-affine adaptation]
compares_with:
  - {target: "[[adachi2025ssa]]", result: better, metric: "R2", note: "UTKFace-C mean 0.619 vs 0.583; SVHN->MNIST 0.473 vs 0.452", locator: "Tables 1-2"}
extends: ["[[adachi2025ssa]]"]
has_limitation:
  - {stated_by: author, note: "Described as a preliminary idea; only two benchmarks; Gaussian assumptions", locator: "Conclusion"}
  - {stated_by: reviewer, note: "Hyperparameter lambda chosen per task without a target-label-free rule; no medical or microscopy evaluation", locator: "Tables 1-2"}
---

# Predictive Spectral Calibration for Source-Free Test-Time Regression

## TL;DR
PSC generalises SSA ([[adachi2025ssa]]): instead of aligning target feature statistics only along the top PCA directions, it matches projected statistics over a fixed bank of probe directions weighted by relevance to the regressor, and adds a term keeping residual energy in the orthogonal complement near a source floor. Only normalisation affine parameters are updated. It improves R2 modestly over SSA on two natural-image regression shifts.

## Citation and link
- Kiet NVT, Trung HT, Hieu PH. arXiv:2603.09338 (10 Mar 2026).
- https://arxiv.org/html/2603.09338

## Problem and motivation
Regression TTA has received little attention; SSA ignores off-subspace drift.

## Data
SVHN->MNIST; UTKFace with 13 corruptions. Depth and head pose mentioned but not evaluated.

## Method
L_PSC = L_support (probe-bank aligned, relevance weighted symmetric KL) + lambda * L_residual. Stored source mean, eigvectors, eigenvalues and residual floor; head frozen.

## Results
- SVHN->MNIST R2: source 0.239, BNA 0.341, SSA 0.452, PSC 0.473 (verified-from-source).
- UTKFace-C mean R2: BNA 0.563, SSA 0.583, PSC 0.619 (verified-from-source).
- The SSA paper reports a different SVHN->MNIST source R2 (0.406), so setups are not identical (reviewer note).

## Limitations
- Author-stated: preliminary, limited benchmarks.
- Reviewer: lambda per task, no rule without target labels; unclear benefit under mild shift.

## Relevance to CD / CV / HEX
Background on regression TTA; our shift (two sources, HEX test gap 0.40 MAPE) is mild, and the gains here are a few R2 points on synthetic corruptions.

## What it suggests we try
Nothing actionable; background only.

## Related notes
- [[adachi2025ssa]] — the method it extends
- [[wang2021tent]] — classification TTA reference

## Sources
- https://arxiv.org/html/2603.09338
