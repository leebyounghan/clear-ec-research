---
id: tanno2019confusion
type: paper
title: "Learning From Noisy Labels By Regularized Estimation Of Annotator Confusion"
authors: [Ryutaro Tanno, Ardavan Saeedi, Swami Sankaranarayanan, Daniel C. Alexander, Nathan Silberman]
year: 2019
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
doi: ""
arxiv: "1902.03680"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/1902.03680"
citation: "Tanno R, Saeedi A, Sankaranarayanan S, Alexander DC, Silberman N. Learning From Noisy Labels By Regularized Estimation Of Annotator Confusion. CVPR 2019:11244-11253. arXiv:1902.03680"
tier: core
group: t6-noisy-regression
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, website]
evidence: peer-reviewed
tldr: "Jointly trains a 'true label' classifier and one confusion matrix per annotator (observed = CM x true), with a trace penalty that makes the split identifiable when annotators are on average diagonally dominant; works even with one label per image and beats EM-based methods."
relevance_clearec: high
relevance_metrics: [CV, HEX, CD]
relevance_note: "Template for 'label = device_transform(true cell statistics) + noise': a true-statistic head plus a separately parameterised device head, with a constraint that keeps the decomposition identifiable."
code_release: unknown
weights_release: none
data_scale: "MNIST and CIFAR-10 with simulated annotators; cardiac ultrasound view classification with sonographer and non-expert labels"
key_numbers:
  - "Observed label distribution of annotator r = CM_r x p(y | x); loss = cross-entropy + trace(CM) penalty; plain gradient descent, CMs initialised to identity (verified-from-source)"
  - "Identifiability theorem: trace penalty recovers true CMs if the mean annotator CM is diagonally dominant (verified-from-source)"
  - "CIFAR-10, one label per image: accuracy 77.65 % (CM error 1.22e-2) vs MBEM 55.97 % vs generalised EM 53.38 % vs vanilla CNN 36.04 % (verified-from-source)"
  - "CIFAR-10, dense labels: 81.23 % vs MBEM 73.33 % (verified-from-source)"
  - "Cardiac view data (real annotators): 75.57 % vs MBEM 73.91 % (verified-from-source)"
  - "Classification only; image-dependent noise and regression left as future work (verified-from-source)"
results:
  - {task: "classification with noisy annotators", dataset: "CIFAR-10 (1 label/image)", metric: "accuracy", value: "77.65 %", comparator: "MBEM", comparator_value: "55.97 %", locator: "CIFAR-10 results table", provenance: "verified-from-source"}
suggests_try: "Regression analogue: a 'true statistic' head T(x) (CV, HEX, CD of the cells in the region) supervised on the 25 expert-marker frames and the 385 Topcon point images, plus a low-capacity device head g(T, image-quality features) fitted on the 8953 device labels; constrain g (monotone, few parameters, e.g. affine in log space with a brightness/vignetting term) so it cannot absorb the image signal, and compare OOF MAPE with the direct regressor."
tags: [annotator-modelling, confusion-matrix, label-noise, identifiability, measurement-model]
search_keywords: [annotator confusion estimation, trace regularisation noisy labels, crowds noisy labels CNN]
uses_dataset: [MNIST, CIFAR-10, cardiac ultrasound views]
uses_method: [confusion-matrix-estimation, trace-regularisation]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Fails when the diagonal-dominance condition is violated; noise assumed independent of the image", locator: "MNIST experiments; Conclusion"}
  - {stated_by: reviewer, note: "Classification CMs; a regression device transform must be parameterised and constrained by us", locator: "Method"}
---

# Learning From Noisy Labels By Regularized Estimation Of Annotator Confusion

## TL;DR
Tanno et al. separate "what the image shows" from "how each annotator distorts it": a classifier predicts the true label distribution and per-annotator confusion matrices map it to each annotator's labels. A trace penalty forces the matrices to absorb as much noise as possible, which provably recovers them under diagonal dominance and works with a single label per image.

## Citation and link
- Tanno R, Saeedi A, Sankaranarayanan S, Alexander DC, Silberman N. CVPR 2019, pp. 11244-11253. arXiv:1902.03680.
- Read: https://ar5iv.labs.arxiv.org/html/1902.03680 ; CVF open-access listing (via search).

## Problem and motivation
With noisy annotators and no gold standard, cross-entropy alone cannot tell annotator error from true-label uncertainty; EM methods need many labels per image.

## Data
MNIST and CIFAR-10 with simulated annotators (pairwise flippers, hammer-spammers; CIFAR annotators about 45 % accurate); cardiac ultrasound view classification with six classes and real expert/non-expert labels (verified-from-source).

## Method
Classifier p(y|x) and annotator CMs; predicted annotator distribution = CM_r p(y|x); cross-entropy to each annotator's labels + lambda * sum trace(CM_r); identity initialisation; gradient descent (verified-from-source).

## Results
- CIFAR-10 dense: 81.23 % (CM error 0.72e-2) vs MBEM 73.33 % vs generalised EM 70.49 % (verified-from-source).
- CIFAR-10 one label/image: 77.65 % vs MBEM 55.97 %, generalised EM 53.38 %, vanilla 36.04 % (verified-from-source).
- Cardiac views: 75.57 % vs 73.91 % (verified-from-source).

## Limitations
- Author-stated: needs average diagonal dominance; image-independent noise.
- Reviewer: no regression version; our "annotator" is a single deterministic device plus sampling noise, so the per-annotator structure reduces to one transform.

## Relevance to CD / CV / HEX
The coordinator's analysis says the remaining CV/HEX error is the device's systematic outline behaviour plus label noise. Facts we have fit a two-part model: device CV is about twice the Voronoi-area CV of expert markers; expert markers read device CV at r 0.89 after scaling and HEX within 1-2 % on clean frames. Tanno's design says: learn the true quantity and the device transform as separate modules and constrain the transform (their trace penalty) so the split is identifiable. With only 25 expert frames, the device head must be tiny (2-4 parameters).

## What it suggests we try
Fit g on the 25 expert frames first (device label vs expert-marker statistics, plus brightness and source ID as covariates) to see whether the device transform is systematic (residual MAPE well below 8.8 on CV) or mostly noise. Only if systematic, add g as a fixed calibration on top of a "true-statistic" head trained on Topcon/expert statistics.

## Related notes
- [[ishida2023bayeserror]] — separate the irreducible floor before modelling the device.
- [[gao2017dldl]] — handling the sampling-noise part.
- [[delange2022mindgap]] — beware corrections that implicitly use the label.

## Sources
- https://ar5iv.labs.arxiv.org/html/1902.03680
- https://openaccess.thecvf.com/content_CVPR_2019/html/Tanno_Learning_From_Noisy_Labels_by_Regularized_Estimation_of_Annotator_Confusion_CVPR_2019_paper.html (via search)
