---
id: wang2019tta
type: paper
title: "Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks"
authors: [Guotai Wang, Wenqi Li, Michael Aertsen, Jan Deprest, Sebastien Ourselin, Tom Vercauteren]
year: 2019
venue: "Neurocomputing"
doi: "10.1016/j.neucom.2019.01.103"
arxiv: "1807.07356"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/1807.07356"
citation: "Wang G, Li W, Aertsen M, Deprest J, Ourselin S, Vercauteren T. Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks. Neurocomputing. 2019;338:34-45. doi:10.1016/j.neucom.2019.01.103"
tier: candidate
group: t7-domain-shift
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: peer-reviewed
tldr: "Formulates test-time augmentation (TTA) as Monte Carlo sampling over acquisition transforms (flip, rotation, scale 0.8-1.2, noise); TTA improves Dice by about 0.7-1.4 points over a single prediction in 2D fetal brain and 3D tumour MRI, beats test-time dropout, and its spread gives fewer overconfident errors."
relevance_clearec: medium
relevance_metrics: [CD, CV, HEX]
relevance_note: "TTA is the safest test-time technique for our regression models: flips and 90-degree rotations leave CD, CV and HEX exactly unchanged, so averaging them is free variance reduction within the 5-min T4 budget; scaling must be excluded for CD."
code_release: open
weights_release: none
data_scale: "2D fetal brain MRI and 3D brain tumour (whole tumour) MRI; test counts not read"
key_numbers:
  - "2D fetal brain, P-Net trained with augmentation: Dice 91.98 single prediction -> 93.05 TTA -> 93.14 TTA+dropout (verified-from-source)"
  - "3D tumour, W-Net trained with augmentation: Dice 88.76 -> 90.07 TTA -> 90.35 TTA+dropout (verified-from-source)"
  - "Test-time dropout alone adds only about 0.05-0.3 Dice; TTA+dropout vs TTA not significant in 2D (verified-from-source)"
  - "Gains plateau beyond N = 20 samples (2D) and N = 40 (3D) (verified-from-source)"
results:
  - {task: "3D brain tumour segmentation", dataset: "BraTS-type whole tumour", metric: "Dice %", value: "90.07 (TTA)", comparator: "single prediction", comparator_value: "88.76", locator: "Table 2", provenance: "verified-from-source"}
suggests_try: "Average every regression and detector prediction over the 8 flip/rot90 views (label-invariant for CD/CV/HEX) plus mild brightness/gamma jitter; test on OOF whether TTA spread flags the outer-decile CV images where regression to the mean is worst."
tags: [test-time-augmentation, uncertainty, aleatoric, medical-segmentation, monte-carlo]
search_keywords: [test-time augmentation uncertainty, aleatoric uncertainty TTA, test-time dropout]
uses_dataset: [fetal brain MRI, BraTS]
uses_method: [test-time augmentation, test-time dropout, Monte Carlo]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Only spatial transforms and noise; intensity changes, elastic deformation and bias fields left as extensions; N must be tuned", locator: "Discussion"}
  - {stated_by: reviewer, note: "Segmentation Dice only; no regression targets; uncertainty-error link shown only qualitatively", locator: "Results"}
---

# Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks

## TL;DR
The paper frames test-time augmentation as sampling plausible acquisitions of the same anatomy and averaging the predictions. Across three 2D and three 3D networks, TTA consistently and significantly raises Dice over a single prediction and over Monte Carlo dropout, and the TTA spread is a better-behaved uncertainty than dropout.

## Citation and link
- Wang G, Li W, Aertsen M, Deprest J, Ourselin S, Vercauteren T. Neurocomputing. 2019;338:34-45. doi:10.1016/j.neucom.2019.01.103 (arXiv 1807.07356)
- https://ar5iv.labs.arxiv.org/html/1807.07356

## Problem and motivation
Segmentation CNNs give no image-based (aleatoric) uncertainty; dropout gives overconfident wrong predictions.

## Data
2D fetal brain MRI; 3D brain tumour MRI (whole tumour). Exact case counts not read.

## Method
Monte Carlo over transforms: flips (p=0.5 per axis), rotation U(0, 2 pi), scale U(0.8, 1.2), Gaussian noise sd 0.05; same priors used for training augmentation. N=20 (2D), N=40 (3D).

## Results
- 2D (trained with augmentation): FCN 92.03 -> 92.79, U-Net 91.93 -> 92.88, P-Net 91.98 -> 93.05 with TTA (verified-from-source).
- 3D (trained with augmentation): W-Net 88.76 -> 90.07, 3D U-Net 88.43 -> 89.41, V-Net 87.44 -> 88.38 (verified-from-source).
- Uncertainty-error relation: qualitative only; correlation not reported.

## Limitations
- Author-stated: limited transform family; N tuning.
- Reviewer: no domain-shift test and no regression outputs.

## Relevance to CD / CV / HEX
For our image-level regressors, flips and rot90 are exact label symmetries (cell areas, neighbour counts and density do not change), so 8-view TTA is a free ensemble; scaling changes CD and must not be used for CD. Under MAPE, TTA variance could also help detect high-uncertainty images; whether it identifies the extreme-CV images (outer deciles hold about a third of CV error) is testable on OOF.

## What it suggests we try
8-view flip/rot90 TTA on all heads and on the point detector (merge points by NMS), measured on OOF MAPE per metric; record TTA spread and check its correlation with absolute error.

## Related notes
- [[adachi2025ssa]] — parameter-updating TTA for regression (riskier)
- [[zhang2020bigaug]] — the same transform families used at training time
- [[lee2022mediar]] — flip TTA + ensemble gave +0.4-0.7 F1 in cell segmentation

## Sources
- https://ar5iv.labs.arxiv.org/html/1807.07356
- https://arxiv.org/abs/1807.07356v3
