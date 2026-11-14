---
id: gao2017dldl
type: paper
title: "Deep Label Distribution Learning With Label Ambiguity"
authors: [Bin-Bin Gao, Chao Xing, Chen-Wei Xie, Jianxin Wu, Xin Geng]
year: 2017
venue: "IEEE Transactions on Image Processing"
doi: "10.1109/TIP.2017.2689998"
arxiv: "1611.01731"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/1611.01731"
citation: "Gao BB, Xing C, Xie CW, Wu J, Geng X. Deep Label Distribution Learning With Label Ambiguity. IEEE Trans Image Process. 2017;26(6):2825-2838. doi:10.1109/TIP.2017.2689998"
tier: core
group: t6-noisy-regression
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, website]
evidence: peer-reviewed
tldr: "Replace each scalar/ordinal label with a discretised Gaussian over neighbouring label values (sigma = known label ambiguity, e.g. annotator std) and train with KL divergence; beats L2/L1 regression and plain classification on age and head-pose estimation."
relevance_clearec: high
relevance_metrics: [CV, HEX]
relevance_note: "Our CV/HEX labels are discrete and carry a known, computable sampling spread from about 150 cells; DLDL is the recipe for encoding that spread as the training target."
code_release: open
weights_release: unknown
data_scale: "Morph, ChaLearn LAP apparent age; Pointing'04, BJUT-3D, AFLW head pose; VOC2007 multi-label; VOC2011 segmentation"
key_numbers:
  - "Target y_j = normalised Gaussian p(l_j | mu, sigma) over discrete labels; loss = KL (cross-entropy) to softmax; decode by expectation or argmax (verified-from-source)"
  - "Morph age MAE: DLDL 2.52 vs L2 regression 3.17 vs L1 regression 2.88 vs softmax classification 2.86 (verified-from-source)"
  - "ChaLearn age MAE / epsilon-error: DLDL 5.34/0.44 vs L1 regression 5.62/0.47 vs classification 7.95/0.58 (verified-from-source)"
  - "sigma = 2 years fixed on Morph (no per-image ambiguity given); ChaLearn provides per-image apparent-age std (verified-from-source)"
  - "MAE vs sigma is V-shaped; authors suggest sigma near the spacing between neighbouring labels, tuned by CV (verified-from-source)"
results:
  - {task: "age estimation", dataset: "Morph", metric: "MAE (years)", value: "2.52", comparator: "L1 regression ConvNet", comparator_value: "2.88", locator: "age results table", provenance: "verified-from-source"}
suggests_try: "Train CV and HEX as label distributions with a per-image sigma from the sampling model (HEX: sqrt(p(1-p)/n) with n about 150, about 7.5 % relative; CV: about 6.5 % of the label), on frozen features; decode the 1/y-weighted median and compare OOF MAPE with the regression head and with a fixed-sigma HL-Gauss head."
tags: [label-distribution-learning, label-ambiguity, ordinal-targets, Gaussian-targets]
search_keywords: [DLDL, deep label distribution learning, Gaussian label distribution age estimation]
uses_dataset: [Morph, ChaLearn LAP 2015, Pointing'04, AFLW]
uses_method: [label-distribution-learning, KL-loss, expectation-decoding]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Expectation decoding was worse than argmax for head pose; sigma must be tuned", locator: "Experiments"}
  - {stated_by: reviewer, note: "Framed as label ambiguity, not label error; no analysis under systematic (biased) label processes", locator: "Introduction"}
---

# Deep Label Distribution Learning With Label Ambiguity

## TL;DR
DLDL turns each label into a discrete Gaussian distribution over nearby label values and trains a CNN with KL divergence. When the label's uncertainty is known per image (ChaLearn apparent-age std) it can be used directly as sigma. It beats regression and classification on age and pose with small datasets.

## Citation and link
- Gao BB, Xing C, Xie CW, Wu J, Geng X. IEEE Trans Image Process. 2017;26(6):2825-2838. doi:10.1109/TIP.2017.2689998.
- Read: https://ar5iv.labs.arxiv.org/html/1611.01731

## Problem and motivation
Labels such as age or pose are ambiguous: neighbouring values are nearly as correct. Hard labels waste this structure and overfit small datasets.

## Data
Age: Morph, ChaLearn LAP (apparent age with per-image mean and std from votes). Head pose: Pointing'04, BJUT-3D, AFLW. Also VOC2007 multi-label and VOC2011 segmentation (verified-from-source).

## Method
- Age: integer labels 1-85; y_j proportional to Gaussian(l_j; mu, sigma); sigma from annotator votes (ChaLearn) or fixed 2 (Morph).
- Pose: 2-D Gaussian, sigma 15 / 5 / 3 degrees per dataset.
- Loss: cross-entropy/KL to softmax; gradient (y_hat - y). Decode: Max (argmax) or Exp (expectation) (verified-from-source).

## Results
- Morph MAE 2.52 (DLDL) vs 3.17 (L2), 2.88 (L1), 2.89 (epsilon-insensitive), 2.86 (classification) (verified-from-source).
- ChaLearn MAE/epsilon-error 5.34/0.44; fine-tuned VGG-Face 3.51/0.31 vs best competition 0.28 with external data (verified-from-source).
- Pointing'04 pose MAE 4.64 deg vs L1 regression 9.43 (verified-from-source).

## Limitations
- Author-stated: sigma sensitivity (V-shaped); Exp decoding not always better.
- Reviewer: no MAPE or relative-error analysis; ambiguity assumed symmetric and centred on the label.

## Relevance to CD / CV / HEX
CV labels have 78 values at step 0.01 and HEX is a share of about 150 cells, so both are discrete with a sampling spread we can compute per image from the label (binomial for HEX, roughly proportional to CV for CV). That is the ChaLearn situation (known per-image sigma), which is the strongest use case of DLDL. It addresses the noise part (b) of the coordinator's split; the device-transform part needs a separate calibration layer (see [[tanno2019confusion]]).

## What it suggests we try
DLDL head on frozen features for HEX first (cleanest sampling model): bins at the label resolution, per-image binomial sigma, KL loss; then CV with sigma = 0.065 x label. Decode the 1/y-weighted median. Keep only if OOF MAPE beats the current head.

## Related notes
- [[farebrother2024hlgauss]] — same idea with a fixed sigma/bin ratio.
- [[kendall2017uncertainties]] — learned instead of known noise level.
- [[gneiting2011point]] — which functional to decode under MAPE.

## Sources
- https://ar5iv.labs.arxiv.org/html/1611.01731
- https://csgaobb.github.io/Projects/DLDL.html (via search, journal citation)
