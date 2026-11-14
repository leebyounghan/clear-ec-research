---
id: tofallis2015logratio
type: paper
title: "A better measure of relative prediction accuracy for model selection and model estimation"
authors: [Chris Tofallis]
year: 2015
venue: "Journal of the Operational Research Society"
doi: "10.1057/jors.2014.103"
arxiv: "2105.05249"
pmcid: ""
url: "https://arxiv.org/abs/2105.05249"
citation: "Tofallis C. A better measure of relative prediction accuracy for model selection and model estimation. J Oper Res Soc. 2015;66(8):1352-1362. doi:10.1057/jors.2014.103"
tier: candidate
group: t6-noisy-regression
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "MAPE systematically favours methods whose predictions are too low; the log accuracy ratio log(prediction/actual) is proposed instead, and least squares on it predicts the geometric mean."
relevance_clearec: medium
relevance_metrics: [CD, CV, HEX]
relevance_note: "Explains why a MAPE-scored challenge rewards slight under-prediction, and gives a log-space training loss whose optimum (geometric mean) is easy to reason about."
code_release: none
weights_release: none
data_scale: "simulations with multiplicative error (sizes not read)"
key_numbers:
  - "MAPE, when used to choose between prediction methods, systematically selects methods whose predictions are too low (from-abstract)"
  - "Least-squares regression on log(prediction/actual) predicts the geometric mean (from-abstract)"
  - "In simulations with multiplicative error the log-ratio criterion is far superior to MAPE for model selection; exact rates not reported in abstract (from-abstract)"
results: []
suggests_try: "If we train in log space (MSE on log y), the model targets the geometric mean; under MAPE we should then shift down by a fitted per-metric factor (OOF), since the MAPE optimum lies below the median, which itself equals the geometric mean only for symmetric log errors."
tags: [MAPE, log-accuracy-ratio, model-selection, under-prediction-bias]
search_keywords: [MAPE bias under-prediction, log accuracy ratio, relative accuracy measure]
uses_dataset: []
uses_method: [log-ratio-least-squares]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Only abstract read (full text paywalled; arXiv copy has no HTML); simulation numbers not checked", locator: "abstract"}
---

# A better measure of relative prediction accuracy for model selection and model estimation

## TL;DR
Tofallis argues MAPE is biased: it prefers models that under-predict. He proposes the log of the accuracy ratio, log(prediction/actual), for both selecting and fitting models; least squares on this quantity estimates the geometric mean.

## Citation and link
- Tofallis C. J Oper Res Soc. 2015;66(8):1352-1362. doi:10.1057/jors.2014.103 (erratum listed by CrossRef, not checked).
- Read: https://arxiv.org/abs/2105.05249 (abstract page; PDF only, no HTML).

## Problem and motivation
Relative accuracy matters when scatter grows with the value (heteroscedastic, multiplicative error). MAPE is the default relative measure in business forecasting, but it penalises over-prediction more than under-prediction of the same ratio.

## Data
Simulations with multiplicative error (design details not read).

## Method
Compare model selection by MAPE vs by log accuracy ratio in simulation; show that least-squares fitting of log Q (Q = prediction/actual) yields geometric-mean predictions.

## Results
- MAPE favours low predictions in model selection (from-abstract).
- Log-ratio criterion far superior in simulations; exact rates not reported in abstract (from-abstract).
- Least squares on log Q predicts the geometric mean (from-abstract).

## Limitations
- Author-stated: not read.
- Reviewer: abstract-level only; the paper argues for changing the metric, which we cannot do (the challenge fixes MAPE), so its use for us is to understand the bias direction.

## Relevance to CD / CV / HEX
Our score is fixed MAPE per metric, so the under-prediction preference Tofallis criticises is something we should exploit, not avoid. If our heads are trained with MSE on log targets, they estimate a geometric mean, which is above the MAPE optimum (median of order -1, see [[gneiting2011point]]).

## What it suggests we try
Fit one multiplicative factor per metric on OOF predictions (grid 0.95-1.01) to minimise MAPE; apply to test. This is a two-minute check and tells us how much "under-prediction bonus" is left in CV/HEX.

## Related notes
- [[demyttenaere2016mape]] — MAPE-optimal = 1/y-weighted median.
- [[gneiting2011point]] — APE elicits the median of order -1.

## Sources
- https://arxiv.org/abs/2105.05249
- https://econpapers.repec.org/RePEc:pal:jorsoc:v:66:y:2015:i:8:p:1352-1362 (search snippet)
