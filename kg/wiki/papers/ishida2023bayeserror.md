---
id: ishida2023bayeserror
type: paper
title: "Is the Performance of My Deep Network Too Good to Be True? A Direct Approach to Estimating the Bayes Error in Binary Classification"
authors: [Takashi Ishida, Ikko Yamane, Nontawat Charoenphakdee, Gang Niu, Masashi Sugiyama]
year: 2023
venue: "International Conference on Learning Representations (ICLR)"
doi: ""
arxiv: "2202.00395"
pmcid: ""
url: "https://arxiv.org/html/2202.00395"
citation: "Ishida T, Yamane I, Charoenphakdee N, Niu G, Sugiyama M. Is the Performance of My Deep Network Too Good to Be True? A Direct Approach to Estimating the Bayes Error in Binary Classification. ICLR 2023. arXiv:2202.00395"
tier: core
group: t6-noisy-regression
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, website]
evidence: peer-reviewed
tldr: "The Bayes (irreducible) error can be estimated without any model by averaging min(c, 1-c) over soft labels c = p(y=1|x); with noisy soft labels the naive estimate is biased and needs correction; ViT already reaches the estimated floor on CIFAR-10H."
relevance_clearec: high
relevance_metrics: [CV, HEX, CD]
relevance_note: "The principle (estimate the floor from the label-generating noise, not from models) is what we need to judge how far CV/HEX/CD OOF error sits above the irreducible label-noise floor; the method itself is binary-classification only."
code_release: open
weights_release: none
data_scale: "CIFAR-10H, Fashion-MNIST-H (new), ICLR 2017-2023 review scores"
key_numbers:
  - "Estimator: mean of min(c_i, 1 - c_i) over clean soft labels; unbiased, consistent, O_P(1/sqrt(n)) (verified-from-source)"
  - "Noisy soft labels: naive estimator is biased and the bias does not vanish with n (Theorem 3.3); corrected with sign labels (verified-from-source)"
  - "Fashion-MNIST-H: estimated Bayes error 3.478 % (95 % CI +-0.079) vs ResNet-18 test error 3.852 % (SE +-0.041) (verified-from-source)"
  - "CIFAR-10H: ViT B/16 error at or slightly above the estimated Bayes error after adjusting for the label distribution (verified-from-source)"
  - "ICLR accept/reject Bayes error estimates 6.8 % (2017) to 9.6 % (2022) (verified-from-source)"
  - "No regression extension in the paper (verified-from-source)"
results:
  - {task: "binary classification floor", dataset: "Fashion-MNIST-H", metric: "error", value: "3.852 % (ResNet-18)", comparator: "estimated Bayes error", comparator_value: "3.478 %", locator: "Fashion-MNIST-H experiment", provenance: "verified-from-source"}
suggests_try: "Regression analogue for us (our derivation): floor MAPE = E|label - truth|/label from the sampling model alone; for Gaussian relative noise with SE 6.5 % (CV), 7.5 % (HEX), 3 % (CD) that is about 0.8 x SE = 5.2, 6.0, 2.4; compare with OOF 10.11 / 9.74 / 6.19 to bound what any device-behaviour model could still gain, and note that, like the paper's noisy soft labels, a floor from an uncertain noise model is itself biased."
tags: [Bayes-error, irreducible-error, label-noise, evaluation]
search_keywords: [Bayes error estimation soft labels, irreducible error deep network, too good to be true]
uses_dataset: [CIFAR-10H, Fashion-MNIST-H]
uses_method: [soft-label-Bayes-error-estimator]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Naive estimator biased under noisy soft labels; human-label distributions may inflate the estimate", locator: "Sec 3, CIFAR-10H discussion"}
  - {stated_by: reviewer, note: "Binary classification only; regression floors need a noise model such as our cell-sampling model", locator: "Method"}
---

# Is the Performance of My Deep Network Too Good to Be True? A Direct Approach to Estimating the Bayes Error in Binary Classification

## TL;DR
Ishida et al. estimate the best achievable error directly from soft labels (class-posterior estimates from multiple annotators), with no model fitting. The estimator is unbiased for clean soft labels, biased for noisy ones unless corrected, and shows that some state-of-the-art classifiers already sit at the floor.

## Citation and link
- Ishida T, Yamane I, Charoenphakdee N, Niu G, Sugiyama M. ICLR 2023. arXiv:2202.00395.
- Read: https://arxiv.org/html/2202.00395 ; ICLR 2023 poster listing (via search).

## Problem and motivation
Without knowing the irreducible error, one cannot tell whether a benchmark has headroom or whether reported results overfit the test set.

## Data
CIFAR-10H (human soft labels), new Fashion-MNIST-H with multiple annotations, ICLR OpenReview scores 2017-2023 (verified-from-source).

## Method
Bayes error = E[min(eta(x), 1 - eta(x))]; plug in soft labels c_i; variants for uncertainty labels, noisy soft labels (bias, sign-label correction), multiple hard labels (bias O(sqrt(log(2nm)/m))), positive-confidence data (verified-from-source).

## Results
- Fashion-MNIST-H: 3.478 % (+-0.079) estimated floor vs 3.852 % ResNet-18 (verified-from-source).
- CIFAR-10H: ViT at or just above the floor after adjusting; below it on raw CIFAR-10 labels (verified-from-source).
- ICLR decisions: floor 6.8-9.6 % across years (verified-from-source).

## Limitations
- Author-stated: noisy soft labels bias the estimate; human labels may inflate it.
- Reviewer: no regression version; applying the idea to us requires our own noise model.

## Relevance to CD / CV / HEX
The coordinator now attributes 78-88 % of remaining CV/HEX error to what even expert centres in the true region miss (device outline behaviour plus label noise). The paper's lesson is to separate the floor that comes from the label-generating process (estimable without a model) from the part a better model could still learn. For us the sampling noise floor is computable from about 150 cells; the device-behaviour part is in principle learnable from the image if it is systematic.

## What it suggests we try
Compute per-metric floors on OOF directly: simulate labels from the sampling model (resample about 150 cells' areas/neighbour counts from expert-marker frames) to get E|label - truth|/label; report floor vs OOF per metric. If OOF minus floor is mostly device behaviour, fit a device-transform layer (see [[tanno2019confusion]]); if it is close to the floor, stop modelling and only tune MAPE decoding.

## Related notes
- [[gneiting2011point]] — with a noise floor, decoding the right functional is what remains.
- [[kendall2017uncertainties]] — per-image aleatoric noise estimates.

## Sources
- https://arxiv.org/html/2202.00395
- https://www.iclr.cc/virtual/2023/poster/10824 (via search)
