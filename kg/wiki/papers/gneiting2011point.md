---
id: gneiting2011point
type: paper
title: "Making and evaluating point forecasts"
authors: [Tilmann Gneiting]
year: 2011
venue: "Journal of the American Statistical Association"
doi: "10.1198/jasa.2011.r10138"
arxiv: "0912.0902"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/0912.0902"
citation: "Gneiting T. Making and evaluating point forecasts. J Am Stat Assoc. 2011;106(494):746-762. doi:10.1198/jasa.2011.r10138"
tier: core
group: t6-noisy-regression
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: review
tldr: "Point forecasts must match the scoring function: squared error elicits the mean, absolute error the median, and absolute percentage error elicits the 'median of order -1' (median of the density proportional to f(y)/y); mismatched scoring can rank a skilled forecaster below a biased one."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Tells us exactly which summary of a predictive distribution to submit under per-metric MAPE: med^(-1), the median of f(y)/y, which lies below the median."
code_release: none
weights_release: none
data_scale: "theory + simulation (n = 100,000)"
key_numbers:
  - "APE |(x-y)/y| is consistent for the median of order -1, med^(-1), i.e. the median of the density proportional to y^(-1) f(y) (verified-from-source)"
  - "Relative error |(x-y)/x| is consistent for the median of order +1, med^(1) (verified-from-source)"
  - "Squared error is consistent for the mean; a score is mean-consistent iff it is a Bregman function (Savage 1971) (verified-from-source)"
  - "Simulation n = 100,000: the 'pessimist' gets the best AE and APE and the 'optimist' the best RE, although the 'statistician' is the most skilled by construction (verified-from-source)"
  - "Simulated APE: statistician 2.58e5 vs pessimist 0.14e5 vs Bayes rule 1.00 (epsilon forecast) (verified-from-source)"
results:
  - {task: "point forecast ranking", dataset: "simulated heteroscedastic process", metric: "APE", value: "pessimist 0.14e5", comparator: "statistician (true model mean)", comparator_value: "2.58e5", locator: "Table 4", provenance: "verified-from-source"}
suggests_try: "Predict a distribution per metric (histogram head or ensemble) and submit med^(-1) = the 1/y-weighted median, not the mean; for a log-normal predictive with log-sd s this equals median*exp(-s^2) (our derivation), i.e. about 2 % below the median at s = 0.15."
tags: [elicitability, scoring-functions, MAPE, consistency]
search_keywords: [elicitability, consistent scoring function, beta-median, MAPE functional]
uses_dataset: []
uses_method: [consistent-scoring-functions, Bayes-rule]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "med^(-1) may not exist for heavy-tailed or near-zero positive variables; then optimal APE forecast tends to zero", locator: "Sec 3 simulation"}
---

# Making and evaluating point forecasts

## TL;DR
A review of scoring functions and elicitability. Each scoring function rewards a specific statistical functional of the forecaster's predictive distribution. Absolute percentage error (the per-sample term of MAPE) rewards the "median of order -1", which is below the median; using the wrong summary can make a skilled forecaster look worse than a biased one.

## Citation and link
- Gneiting T. J Am Stat Assoc. 2011;106(494):746-762. doi:10.1198/jasa.2011.r10138.
- Read: https://ar5iv.labs.arxiv.org/html/0912.0902 (arXiv version).

## Problem and motivation
Forecast competitions and papers rank point forecasts by an error measure without saying which functional (mean, median, quantile) forecasters should report. Gneiting shows this leads to "grossly misguided" comparisons unless the score and the functional match.

## Data
Theory, plus a simulation of a heteroscedastic process with n = 100,000 forecast cases and four forecasters (statistician, optimist, pessimist, Mr. Bayes) (verified-from-source).

## Method
- Consistency: score S is consistent for functional T if the expected score is minimised by reporting T(F).
- Beta-median: median of a variable whose density is proportional to y^beta f(y); score |1 - (y/x)^beta| is consistent for it (via the weighted-score theorem).
- APE = beta = -1 case; relative error |x-y|/x = beta = +1 case.
- Quantiles: generalised piecewise-linear scores; mean: Bregman scores.

## Results
- APE consistent for med^(-1) (verified-from-source).
- Table 4 (n = 100,000): SE / AE / APE / RE = statistician 5.07 / 0.97 / 2.58e5 / 0.97; optimist 22.73 / 4.35 / 13.96e5 / 0.87; pessimist 7.61 / 0.96 / 0.14e5 / 19.24; Bayes rule per score 5.07 / 0.86 / 1.00 / 0.75 (verified-from-source).
- Central advice: specify the scoring function ex ante or name an elicitable functional (verified-from-source).

## Limitations
- Author-stated: med^(-1) may fail to exist for heavy-tailed positive variables; then the APE-optimal forecast is an arbitrarily small value.
- Reviewer: no worked guidance on how far med^(-1) sits from the median for light-tailed, low-noise targets like ours.

## Relevance to CD / CV / HEX
The CLEAR-EC score is MAPE, so the target functional is med^(-1) of p(label | image). With the label noise we know (about 6.5 % SE on CV, 7.5 % on HEX, 3 % on CD from about 150 cells) plus model uncertainty, p(label | image) has a relative spread of roughly 10-15 % for CV/HEX, so med^(-1) is a few percent below the median. The paper also implies that a well-calibrated regression mean that "regresses to the mean" is the correct target for squared error, not for MAPE; any de-shrinking must be judged against med^(-1), not against matching the label spread.

## What it suggests we try
Add a histogram (78-bin for CV, step 0.01) or quantile head per metric and decode the 1/y-weighted median; alternatively apply a per-metric multiplicative factor c < 1 fitted on OOF to minimise MAPE (a cheap proxy for med^(-1)). Report OOF MAPE for mean, median and 1/y-weighted median decoding.

## Related notes
- [[demyttenaere2016mape]] — same result from the learning-theory side (weighted MAE, weights 1/y).
- [[tofallis2015logratio]] — MAPE's asymmetry and the log-ratio alternative.
- [[stewart2023regclass]] — histogram heads that give a full predictive distribution to decode.

## Sources
- https://ar5iv.labs.arxiv.org/html/0912.0902
- https://econpapers.repec.org/RePEc:bes:jnlasa:v:106:i:494:y:2011:p:746-762 (search snippet with DOI)
