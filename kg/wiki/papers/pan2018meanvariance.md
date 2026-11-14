---
id: pan2018meanvariance
type: paper
title: "Mean-Variance Loss for Deep Age Estimation From a Face"
authors: [Hongyu Pan, Hu Han, Shiguang Shan, Xilin Chen]
year: 2018
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
doi: ""
arxiv: ""
pmcid: ""
url: "https://openaccess.thecvf.com/content_cvpr_2018/html/Pan_Mean-Variance_Loss_for_CVPR_2018_paper.html"
citation: "Pan H, Han H, Shan S, Chen X. Mean-Variance Loss for Deep Age Estimation From a Face. CVPR 2018:5285-5294"
tier: candidate
group: t6-noisy-regression
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "Age estimation as distribution learning over discrete ages: softmax loss plus a mean loss (distribution mean vs true age) and a variance loss that keeps the predicted distribution concentrated; reported to beat prior single models on FG-NET, MORPH II and CLAP2016."
relevance_clearec: low
relevance_metrics: [CV, HEX]
relevance_note: "Distributional head for discrete, ambiguous targets; but its variance penalty forces sharp distributions, which conflicts with using the predictive spread for MAPE-optimal decoding under our label noise."
code_release: unknown
weights_release: unknown
data_scale: "FG-NET, MORPH Album II, CLAP2016 (sizes not read)"
key_numbers:
  - "Loss = softmax + mean loss (squared gap between distribution mean and true age) + variance loss (variance of predicted distribution), trained end to end with SGD (from-abstract)"
  - "Reported to outperform state of the art by a large margin with a single model; exact MAE not reported in abstract (from-abstract)"
  - "ChaLearn/CLAP2016 epsilon-error 0.2867 per a benchmark listing (from-abstract)"
results: []
suggests_try: "If we add a histogram/DLDL head for CV or HEX, do not add a variance penalty: with about 6.5-7.5 % irreducible label noise the predictive spread should stay wide, and it is needed to decode the 1/y-weighted median; use mean-variance only as a comparison arm if decoding by expectation."
tags: [distribution-learning, age-estimation, mean-variance-loss]
search_keywords: [mean-variance loss, age estimation distribution learning]
uses_dataset: [FG-NET, MORPH II, CLAP2016]
uses_method: [mean-variance-loss, softmax-expectation]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Abstract-level only (CVF page returned 403); numbers not checked", locator: "n/a"}
  - {stated_by: reviewer, note: "Variance penalty yields overconfident distributions when labels are noisy", locator: "Method"}
---

# Mean-Variance Loss for Deep Age Estimation From a Face

## TL;DR
Pan et al. predict a softmax distribution over ages and add two terms: the distribution's mean should match the label and its variance should be small. This distribution-learning loss beat earlier age estimators on three benchmarks.

## Citation and link
- Pan H, Han H, Shan S, Chen X. CVPR 2018, pp. 5285-5294.
- Read: abstract via search listings (CVF open-access page returned 403).

## Problem and motivation
Exact regression ignores the ambiguity of age labels; a distribution over ages is more robust.

## Data
FG-NET (leave-one-person-out), MORPH Album II, CLAP2016 (from-abstract / search snippet).

## Method
Softmax over discrete ages + mean loss + variance loss; expectation decoding; end-to-end SGD (from-abstract).

## Results
Large-margin improvement claimed; exact MAE not read. CLAP2016 epsilon-error 0.2867 per a benchmark listing (from-abstract).

## Limitations
- Author-stated: not read.
- Reviewer: abstract only; concentrating the distribution is the opposite of what label-noise-aware decoding needs.

## Relevance to CD / CV / HEX
Relevant only as a contrast to [[gao2017dldl]] and [[farebrother2024hlgauss]]: for CV/HEX the label noise is a known, irreducible spread (about 150 cells), so a well-calibrated predictive distribution should not be shrunk; under MAPE we decode a functional that depends on that spread ([[gneiting2011point]]).

## What it suggests we try
Nothing new; if a distributional head is built, keep the spread (no variance penalty) and decode the 1/y-weighted median.

## Related notes
- [[gao2017dldl]] — Gaussian label distributions with known sigma.
- [[kendall2017uncertainties]] — learned per-image spread.

## Sources
- https://openaccess.thecvf.com/content_cvpr_2018/html/Pan_Mean-Variance_Loss_for_CVPR_2018_paper.html (403; abstract via search)
