---
id: delange2022mindgap
type: paper
title: "Mind the gap: Performance metric evaluation in brain-age prediction"
authors: [Ann-Marie G. de Lange, Melis Anatürk, Jaroslav Rokicki, Laura K. M. Han, Katja Franke, Dag Alnæs, et al.]
year: 2022
venue: "Human Brain Mapping"
doi: "10.1002/hbm.25837"
arxiv: ""
pmcid: "PMC9188975"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC9188975"
citation: "de Lange AMG, Anatürk M, Rokicki J, Han LKM, Franke K, Alnæs D, et al. Mind the gap: Performance metric evaluation in brain-age prediction. Hum Brain Mapp. 2022;43(10):3113-3129. doi:10.1002/hbm.25837"
tier: core
group: t6-noisy-regression
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "Regression predictions are always biased toward the training mean when the input-target relation is imperfect; 'age-bias corrections' that use the true target make even poor models look accurate, while the label-free slope correction just inflates prediction variance."
relevance_clearec: high
relevance_metrics: [CV, HEX, CD]
relevance_note: "Our CV predictions span 66 % of the label spread at r 0.686; this paper says that compression is the normal signature of an imperfect regressor and that label-free de-shrinking inflates variance rather than fixing error."
code_release: unknown
weights_release: none
data_scale: "UK Biobank N = 37,156; Cam-CAN N = 622 (T1 MRI, 10-fold CV)"
key_numbers:
  - "UKB, 10-fold CV: r 0.728 -> 0.898, R2 0.529 -> 0.760, MAE 4.140 -> 2.969 years after age-bias correction that uses chronological age (verified-from-source)"
  - "Cam-CAN: r 0.870 -> 0.927, R2 0.753 -> 0.837, MAE 7.403 -> 6.001 years after the same correction (verified-from-source)"
  - "Age-bias corrected metrics indicate high accuracy even for poorly performing models (shuffled-label experiments) (verified-from-source)"
  - "The label-free slope correction (divide by slope, Cole et al. 2018) inflates variance in the data (verified-from-source)"
  - "r and R2 are lower in samples with a narrower target range; MAE/RMSE fall with a narrower range (verified-from-source)"
results:
  - {task: "brain-age regression", dataset: "UK Biobank", metric: "MAE (years)", value: "2.969 (corrected with true age)", comparator: "uncorrected", comparator_value: "4.140", locator: "Table 3", provenance: "verified-from-source"}
suggests_try: "Do not stretch CV/HEX predictions to match the label spread: for a calibrated predictor the prediction/label SD ratio equals r (0.686 vs our observed 0.66, our derivation), so the compression is expected; any de-shrink factor must be fitted on OOF MAPE, and the 'outer deciles' diagnostic (binned by true label) will always look bad."
tags: [regression-to-the-mean, bias-correction, evaluation-pitfalls, shrinkage]
search_keywords: [brain age bias correction, regression toward the mean predictions, age-bias corrected metrics]
uses_dataset: [UK Biobank, Cam-CAN]
uses_method: [age-bias-correction, slope-correction]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Corrected metrics can hide low prediction variance; recommend inspecting uncorrected results", locator: "Discussion"}
  - {stated_by: reviewer, note: "Does not evaluate label-free corrections on held-out MAE, so the cost of variance inflation is argued, not measured", locator: "Methods"}
---

# Mind the gap: Performance metric evaluation in brain-age prediction

## TL;DR
Brain-age models predict chronological age from MRI; their predictions are pulled toward the training mean (over-estimate the young, under-estimate the old). Common "age-bias corrections" regress the prediction on the true age and correct each subject using that true age; this makes metrics look excellent even for near-random models. The label-free variant (dividing by the fitted slope) inflates variance.

## Citation and link
- de Lange AMG, et al. Hum Brain Mapp. 2022;43(10):3113-3129. doi:10.1002/hbm.25837.
- Read: https://pmc.ncbi.nlm.nih.gov/articles/PMC9188975

## Problem and motivation
Studies compare brain-age models by r, R2, MAE across cohorts with different age ranges and often after bias correction, so reported numbers are not comparable and can be misleading.

## Data
UK Biobank (N = 37,156) and Cam-CAN (N = 622) T1-weighted MRI, 10-fold CV; additional experiments with 0-75 % shuffled labels and restricted age ranges (verified-from-source).

## Method
- Fit predicted vs chronological age (Y = alpha*Omega + beta) and correct each prediction with that subject's chronological age (needs the test label).
- Discusses Cole et al. 2018 slope correction: (prediction - intercept) / slope, which needs no label but "inflates variance".
- Compare r, R2, MAE, RMSE before and after correction, across shuffling levels and age ranges.

## Results
- UKB: r 0.728 -> 0.898; R2 0.529 -> 0.760; MAE 4.140 -> 2.969 years (verified-from-source).
- Cam-CAN: r 0.870 -> 0.927; R2 0.753 -> 0.837; MAE 7.403 -> 6.001 years (verified-from-source).
- Poorest (shuffled) models show the largest improvement after correction; numbers only in figures (verified-from-source).
- Narrow age ranges lower r and R2; R2 becomes negative in the narrowest test set (65-82 years) (verified-from-source).

## Limitations
- Author-stated: corrected metrics conceal low prediction variance; report uncorrected metrics.
- Reviewer: brain-age targets are noise-free (chronological age), unlike our noisy device labels; with label noise the case against de-shrinking is even stronger.

## Relevance to CD / CV / HEX
Our diagnostic "predictions span only 66 % of the CV label spread, outer deciles hold a third of the error" is the same age-bias picture. Binning errors by the true label always shows over-prediction at the low end and under-prediction at the high end for any imperfect regressor, so it does not by itself prove a fixable defect. A label-free stretch (divide by slope) is the Cole correction, which this paper says inflates variance. With labels themselves noisy (about 6.5 % SE on CV), part of the label spread is noise that no predictor should follow.

## What it suggests we try
Treat de-shrinking as a hyperparameter: fit y_hat' = m + k (y_hat - m) on OOF with k in [0.9, 1.3] by OOF MAPE per metric, nested across folds; adopt only if nested OOF MAPE improves. Bin errors by prediction (not by label) to check calibration: a calibrated model should show near-zero mean error in each prediction bin.

## Related notes
- [[yang2021imbalanced]] — tail-range reweighting, the "train-time" alternative to post-hoc stretching.
- [[gneiting2011point]] — the MAPE optimum is a functional of p(y | x), which is itself shrunk.

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9188975
