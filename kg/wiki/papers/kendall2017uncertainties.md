---
id: kendall2017uncertainties
type: paper
title: "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
authors: [Alex Kendall, Yarin Gal]
year: 2017
venue: "Advances in Neural Information Processing Systems 30 (NIPS 2017)"
doi: ""
arxiv: "1703.04977"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/1703.04977"
citation: "Kendall A, Gal Y. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? Advances in Neural Information Processing Systems 30 (NIPS 2017). 2017:5574-5584. arXiv:1703.04977"
tier: core
group: t6-noisy-regression
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, website]
evidence: peer-reviewed
tldr: "Heteroscedastic regression loss 0.5*exp(-s)*||y - f||^2 + 0.5*s (s = predicted log-variance) lets the network attenuate the loss on noisy targets; it gives small gains on depth regression and segmentation, and aleatoric variance does not shrink with more data."
relevance_clearec: high
relevance_metrics: [CV, HEX, CD]
relevance_note: "Gives a per-image predicted noise level, which both down-weights noisy labels in training and tells us how far to shift each prediction toward the MAPE-optimal point (median of order -1)."
code_release: none
weights_release: none
data_scale: "CamVid, NYUv2 (40-class and depth), Make3D"
key_numbers:
  - "Loss = 0.5*exp(-s)*||y - y_hat||^2 + 0.5*s with s = log sigma^2 ('learned loss attenuation'); Laplace/L1 variant derived and usually better for vision regression (verified-from-source)"
  - "NYUv2 depth rel/rms/log10: baseline 0.117/0.517/0.051 vs +aleatoric 0.112/0.508/0.046 vs +both 0.110/0.506/0.045 (verified-from-source)"
  - "Make3D rel: baseline 0.167 vs +aleatoric 0.149 (verified-from-source)"
  - "CamVid IoU: baseline 67.1 vs +aleatoric 67.4 vs +both 67.5 (verified-from-source)"
  - "Aleatoric uncertainty cannot be explained away with more data; its variance stayed roughly constant as training size fell, epistemic rose (verified-from-source)"
results:
  - {task: "monocular depth", dataset: "NYUv2", metric: "rel", value: "0.112 (+aleatoric)", comparator: "baseline", comparator_value: "0.117", locator: "depth results table", provenance: "verified-from-source"}
suggests_try: "Add a log-variance output to the CV and HEX heads with the Laplace (L1) form on log-labels; use the predicted per-image scale s_i to decode the MAPE optimum (for log-normal error, median*exp(-s_i^2), our derivation) so noisier images are pulled down more, and check on OOF whether predicted s_i tracks the absolute OOF error."
tags: [aleatoric-uncertainty, heteroscedastic-loss, loss-attenuation, noisy-targets]
search_keywords: [heteroscedastic aleatoric loss, learned loss attenuation, Kendall Gal uncertainty]
uses_dataset: [CamVid, NYUv2, Make3D]
uses_method: [heteroscedastic-regression, MC-dropout, Laplace-likelihood]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Aleatoric models alone do not capture model (epistemic) uncertainty; out-of-distribution inputs need epistemic terms", locator: "Discussion"}
  - {stated_by: reviewer, note: "Gains in mean error are small; the main value is the uncertainty output, not accuracy", locator: "Results"}
---

# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

## TL;DR
Kendall and Gal separate aleatoric (data/label noise) from epistemic (model) uncertainty and give a simple heteroscedastic loss in which the network predicts a per-input variance that scales down the residual of noisy samples. Accuracy improves slightly on depth regression and segmentation; aleatoric noise does not decrease with more data.

## Citation and link
- Kendall A, Gal Y. NIPS 2017, Advances in Neural Information Processing Systems 30, pp. 5574-5584. arXiv:1703.04977.
- Read: https://ar5iv.labs.arxiv.org/html/1703.04977 ; NeurIPS proceedings listing (via search).

## Problem and motivation
Vision models output point estimates without saying when inputs or labels are inherently noisy; Bayesian deep learning mostly modelled only epistemic uncertainty.

## Data
CamVid and NYUv2 40-class segmentation; NYUv2 and Make3D depth regression (verified-from-source).

## Method
- Aleatoric: network outputs mean and s = log sigma^2; loss 0.5 exp(-s) ||y - y_hat||^2 + 0.5 s (verified-from-source).
- Laplace prior gives an L1 residual with learned scale; authors usually find it better than L2 for vision regression (verified-from-source).
- Epistemic: MC dropout; combined model uses both.

## Results
- NYUv2 depth rel/rms/log10: 0.117/0.517/0.051 -> 0.112/0.508/0.046 (aleatoric) -> 0.110/0.506/0.045 (both) (verified-from-source).
- Make3D rel 0.167 -> 0.149 (aleatoric) (verified-from-source).
- CamVid IoU 67.1 -> 67.4 (aleatoric) -> 67.5 (both); NYUv2 seg 36.5 -> 37.1 -> 37.3 (verified-from-source).
- Aleatoric variance roughly constant as data shrinks; epistemic grows (verified-from-source).

## Limitations
- Author-stated: aleatoric-only models miss epistemic uncertainty.
- Reviewer: the variance head is not calibrated by construction; needs a check on held-out data.

## Relevance to CD / CV / HEX
The coordinator's root-cause analysis says most remaining CV/HEX error is label noise plus device outline behaviour, i.e. aleatoric from the image's point of view. Kendall-Gal says that error will not fall with more data or better features, but a per-image noise estimate is still useful: (1) training down-weights frames whose labels are least predictable (dark, vignetted 2023-24 frames, atypical regions); (2) under MAPE the optimal point is below the median by an amount that grows with the predictive spread, so a per-image spread gives a per-image shift instead of one global factor.

## What it suggests we try
On frozen features: head with outputs (mu_i, b_i), Laplace loss |log y - mu_i| / b_i + log b_i for CV and HEX; decode y_hat = exp(mu_i) * f(b_i) with f fitted on OOF (one parameter); compare OOF MAPE with the global-factor version. Check calibration: corr(b_i, |OOF log error|).

## Related notes
- [[gneiting2011point]] — MAPE optimum depends on the predictive distribution's spread.
- [[gao2017dldl]] — distributional head with a fixed, known label spread instead of a learned one.
- [[farebrother2024hlgauss]] — histogram alternative.

## Sources
- https://ar5iv.labs.arxiv.org/html/1703.04977
- https://papers.nips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html (via search)
