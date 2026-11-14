---
id: stewart2023regclass
type: paper
title: "Regression as Classification: Influence of Task Formulation on Neural Network Features"
authors: [Lawrence Stewart, Francis Bach, Quentin Berthet, Jean-Philippe Vert]
year: 2023
venue: "International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 206"
doi: ""
arxiv: "2211.05641"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/2211.05641"
citation: "Stewart L, Bach F, Berthet Q, Vert JP. Regression as Classification: Influence of Task Formulation on Neural Network Features. AISTATS 2023, PMLR 206:11563-11582. arXiv:2211.05641"
tier: candidate
group: t6-noisy-regression
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html]
evidence: peer-reviewed
tldr: "For two-layer ReLU networks, square-loss regression and cross-entropy over binned targets learn different feature supports; in synthetic 1-D experiments every classification model beat the best regression model."
relevance_clearec: low
relevance_metrics: [CV, HEX]
relevance_note: "Theoretical backing for replacing MSE heads with binned classification heads; evidence is synthetic only."
code_release: unknown
weights_release: none
data_scale: "synthetic 1-D (250 train / 250 val) and 2-D (625 grid points) teacher-network data"
key_numbers:
  - "1-D validation RMSE x100, best/worst/mean: regression 3.70/6.85/4.55 vs classification 0.86/1.54/1.21 over 30 initialisations (verified-from-source)"
  - "k = 50 uniform bins in 1-D, k = 25 in 2-D; prediction = expectation over bins (verified-from-source)"
  - "Regression feature support had 18 points vs 230 of 250 training points for classification (verified-from-source)"
  - "Three-layer networks did not show the same regression under-fitting (10 runs) (verified-from-source)"
results:
  - {task: "1-D synthetic regression", dataset: "teacher network, 9 neurons", metric: "val RMSE x100 (mean)", value: "1.21 (classification)", comparator: "square-loss regression", comparator_value: "4.55", locator: "Table 1", provenance: "verified-from-source"}
suggests_try: "Weak support only: use it to justify trying a binned CV head (see farebrother2024hlgauss), not as evidence it will help a deep pretrained backbone."
tags: [regression-as-classification, feature-learning, theory]
search_keywords: [regression as classification, binned regression, cross-entropy vs square loss]
uses_dataset: []
uses_method: [binned-cross-entropy]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Two-layer networks and 1-D theory only; deeper nets did not show the effect", locator: "Appendix G"}
---

# Regression as Classification: Influence of Task Formulation on Neural Network Features

## TL;DR
The authors analyse two-layer ReLU nets as measures over feature space and show square loss and binned cross-entropy induce different optimal feature supports; regression supports are sparse and harder to optimise. Synthetic experiments show classification heads are more accurate and more stable.

## Citation and link
- Stewart L, Bach F, Berthet Q, Vert JP. AISTATS 2023, PMLR 206:11563-11582.
- Read: https://ar5iv.labs.arxiv.org/html/2211.05641

## Problem and motivation
Practitioners often bin continuous targets and train with cross-entropy (age estimation, depth, pose); the paper asks why this can beat direct regression.

## Data
Synthetic teacher-network targets: 1-D (250 train / 250 val, k = 50 bins), 2-D (625 grid points, k = 25) (verified-from-source).

## Method
Compare square-loss regression (10,000 hidden units) with cross-entropy on bins (500 hidden units), gradient descent, early stopping, 30 seeds; prediction = expected bin value.

## Results
- 1-D val RMSE x100: regression mean 4.55 (best 3.70) vs classification mean 1.21 (best 0.86) (verified-from-source).
- Every classification model beat the best regression model (verified-from-source).

## Limitations
- Author-stated: 1-D theory, two-layer nets; three-layer nets did not under-fit.
- Reviewer: no label noise, no real data; label smoothing and median decoding not covered.

## Relevance to CD / CV / HEX
Background for the binned-head idea on CV/HEX, whose labels are discrete; the decisive practical evidence must come from our own OOF test.

## What it suggests we try
Nothing actionable on its own; see [[farebrother2024hlgauss]] for the recipe.

## Related notes
- [[farebrother2024hlgauss]] — practical Gaussian-smoothed histogram loss.
- Cao et al. 2020 (CORAL) — ordinal formulation of discrete targets; no note in this KG.

## Sources
- https://ar5iv.labs.arxiv.org/html/2211.05641
- https://proceedings.mlr.press/v206/stewart23a.html (via search)
