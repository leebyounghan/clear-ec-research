---
id: demyttenaere2016mape
type: paper
title: "Mean Absolute Percentage Error for regression models"
authors: [Arnaud de Myttenaere, Boris Golden, Bénédicte Le Grand, Fabrice Rossi]
year: 2016
venue: "Neurocomputing"
doi: "10.1016/j.neucom.2015.12.114"
arxiv: "1605.02541"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/1605.02541"
citation: "de Myttenaere A, Golden B, Le Grand B, Rossi F. Mean Absolute Percentage Error for regression models. Neurocomputing. 2016;192:38-48. doi:10.1016/j.neucom.2015.12.114"
tier: core
group: t6-noisy-regression
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: peer-reviewed
tldr: "Proves that minimising MAPE is weighted MAE regression with weights 1/|y|, that a MAPE-optimal model exists and that MAPE empirical risk minimisation is universally consistent; the MAPE-optimal estimate sits below the conditional median."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Our score is per-metric MAPE, so the loss to train/decode is MAE weighted by 1/y, and the optimal point is a 1/y-weighted median, not the mean or median."
code_release: none
weights_release: none
data_scale: "theory + 1-D simulation (1000 train / 1000 test points)"
key_numbers:
  - "Minimising MAPE is equivalent to weighted MAE regression with sample weights 1/|y_i| (verified-from-source)"
  - "Kernel version = weighted quantile (pinball) regression with weights 1/|y_i| (verified-from-source)"
  - "Consistency proof requires |Y| >= Y_L > 0 almost surely (labels bounded away from zero) (verified-from-source)"
  - "Simulation: 1000 training and 1000 test points, Gaussian kernel, C chosen by 5-fold CV (verified-from-source)"
  - "MAPE-trained model clearly beats MAE-trained model only when targets are near zero; for targets far from zero the two tie (verified-from-source)"
results:
  - {task: "1-D kernel regression", dataset: "sinc-type simulation", metric: "test MAPE", value: "lower than MAE-model only for small offsets a", comparator: "MAE-trained model", comparator_value: "equal for large a", locator: "Sec 6.3, Table 1", provenance: "verified-from-source"}
suggests_try: "Train or post-hoc decode each metric with 1/y-weighted L1 (or 1/y-weighted median over an ensemble / predicted distribution) instead of MSE/plain L1; expected gain is small for CD (labels far from zero, low relative noise) and largest where relative label spread is big (CV 0.29-0.50)."
tags: [MAPE, loss-metric-alignment, weighted-median, elicitability]
search_keywords: [MAPE optimal estimator, weighted MAE, weighted median 1/y]
uses_dataset: []
uses_method: [weighted-MAE, kernel-quantile-regression]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Consistency needs labels bounded away from zero; regularised ERM consistency left open", locator: "Sec 5-7"}
  - {stated_by: reviewer, note: "Only a 1-D simulation; no deep networks, no label noise model", locator: "Sec 6"}
---

# Mean Absolute Percentage Error for regression models

## TL;DR
The paper formalises regression under MAPE. Minimising expected MAPE is exactly a weighted MAE problem with weights 1/|y|, so the pointwise optimum is a weighted median of the conditional label distribution (weights 1/y), which is below the ordinary median. ERM under MAPE is universally consistent when labels are bounded away from zero.

## Citation and link
- de Myttenaere A, Golden B, Le Grand B, Rossi F. Neurocomputing. 2016;192:38-48. doi:10.1016/j.neucom.2015.12.114 (ESANN 2015 selected papers).
- Read: https://ar5iv.labs.arxiv.org/html/1605.02541 ; https://arxiv.org/abs/1605.02541v2

## Problem and motivation
MAPE is widely used to report regression quality but models are usually trained with MSE/MAE. The authors ask what the MAPE-optimal model is and whether minimising empirical MAPE is a sound learning principle.

## Data
Theory, plus a 1-D sinc-type simulation (X ~ U[-1,1], heteroscedastic Gaussian noise, translation offset a moving targets away from zero), 1000 train / 1000 test points (verified-from-source).

## Method
- Pointwise optimum m*(x) = argmin_m E(|m - Y| / |Y| | X = x); existence when P(Y = 0) = 0 plus summability conditions; convex and coercive loss (verified-from-source).
- When the minimiser is an interval, the midpoint is used by convention.
- Consistency theorem for ERM over growing classes with finite VC dimension, assuming |Y| >= Y_L > 0.
- Practical recipe: any MAE learner with sample weights 1/|y_i|; for kernels, weighted quantile regression.

## Results
- MAPE optimum = weighted MAE with weights 1/|y| (verified-from-source).
- The MAPE-optimal estimator is "closer to 0" than the conditional median (Sec 6.3.3, graphical) (verified-from-source).
- In simulation the MAPE-trained model wins clearly only for small offsets a (targets near zero); for large a the two models tie (verified-from-source). Exact table values: extraction was garbled, not reproduced here.

## Limitations
- Author-stated: requires labels bounded away from zero; regularised ERM consistency not proved.
- Reviewer: no label-noise analysis; no deep-learning experiments; the below-median shift is shown, not quantified.

## Relevance to CD / CV / HEX
All three CLEAR-EC labels are strictly positive and far from zero (CV 0.29-0.50, HEX share, CD in thousands), so the theory applies cleanly. The prediction to submit per image is the 1/y-weighted median of p(y | image), not E[y | image]. Because the shift below the median grows with the relative spread of p(y | image), it matters most for CV and HEX (relative sampling SE about 6.5 % and 7.5 % from about 150 cells) and least for CD (about 3 %).

## What it suggests we try
Replace MSE/L1 heads with 1/y-weighted L1 per metric (or keep the current models and decode the 1/y-weighted median of the fold/seed ensemble or of a predicted histogram). Check on OOF: compare MAPE of mean vs median vs 1/y-weighted median of ensemble members per metric. Expect small CD gain, a little more for CV/HEX.

## Related notes
- [[gneiting2011point]] — names the functional MAPE elicits (median of order -1).
- [[tofallis2015logratio]] — argues MAPE favours under-prediction; log accuracy ratio alternative.
- [[stewart2023regclass]] — distributional head that makes weighted-median decoding easy.

## Sources
- https://ar5iv.labs.arxiv.org/html/1605.02541
- https://arxiv.org/abs/1605.02541v2
