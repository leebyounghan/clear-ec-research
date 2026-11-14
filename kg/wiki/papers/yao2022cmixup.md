---
id: yao2022cmixup
type: paper
title: "C-Mixup: Improving Generalization in Regression"
authors: [Huaxiu Yao, Yiping Wang, Linjun Zhang, James Zou, Chelsea Finn]
year: 2022
venue: "Advances in Neural Information Processing Systems (NeurIPS)"
doi: ""
arxiv: "2210.05775"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/2210.05775"
citation: "Yao H, Wang Y, Zhang L, Zou J, Finn C. C-Mixup: Improving Generalization in Regression. NeurIPS 2022. arXiv:2210.05775"
tier: candidate
group: t6-noisy-regression
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html, website]
evidence: peer-reviewed
tldr: "Mixup for regression that picks mixing partners with a Gaussian kernel on label distance, so interpolated labels stay plausible; small but consistent gains in RMSE and MAPE, including on echocardiogram ejection-fraction regression."
relevance_clearec: medium
relevance_metrics: [CV, HEX, CD]
relevance_note: "Cheap regulariser for image regression heads reported in MAPE on a medical video task; label-aware pairing is natural for our scalar CD/CV/HEX targets."
code_release: open
weights_release: unknown
data_scale: "Airfoil, NO2, Exchange-rate, Electricity, EchoNet echocardiograms (Echo); OOD: RCF-MNIST, PovertyMap, Crime, SkillCraft, DTI"
key_numbers:
  - "Echo (ejection fraction) RMSE/MAPE: ERM 5.402/8.700 % vs mixup 5.393/8.838 % vs C-Mixup 5.177/8.435 % (verified-from-source)"
  - "NO2 RMSE/MAPE: ERM 0.537/13.615 % vs C-Mixup 0.509/12.998 % (verified-from-source)"
  - "Abstract: +6.56 % in-distribution, +4.76 % task generalisation, +5.82 % OOD robustness vs best prior approach (verified-from-source)"
  - "Partner sampling: P(j | i) proportional to exp(-||y_i - y_j||^2 / (2 sigma^2)) (verified-from-source)"
results:
  - {task: "ejection-fraction regression", dataset: "EchoNet (Echo)", metric: "MAPE", value: "8.435 %", comparator: "ERM", comparator_value: "8.700 %", locator: "Table 1", provenance: "verified-from-source"}
suggests_try: "If any end-to-end fine-tuning run remains, add C-Mixup on images with partners sampled by label distance (separately per metric, or by CV label which has the largest gap); expected gain is a few percent relative at best, so only worth it as a free add-on to a run we already do."
tags: [mixup, regression, augmentation, generalisation]
search_keywords: [C-Mixup, label-aware mixup regression]
uses_dataset: [EchoNet-Dynamic, Airfoil, NO2, PovertyMap]
uses_method: [mixup, kernel-partner-sampling]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Mixing two specular images does not produce a physically valid cell mosaic whose CV/HEX is the label mix; gains for morphometric targets are uncertain", locator: "Method"}
---

# C-Mixup: Improving Generalization in Regression

## TL;DR
Vanilla mixup mixes random pairs, which creates wrong labels for regression. C-Mixup samples partners with probability decaying in label distance, yielding modest gains in-distribution, across tasks and under shift.

## Citation and link
- Yao H, Wang Y, Zhang L, Zou J, Finn C. NeurIPS 2022. arXiv:2210.05775.
- Read: https://ar5iv.labs.arxiv.org/html/2210.05775 ; NeurIPS poster page (search listing).

## Problem and motivation
Mixup assumes linear label interpolation is valid; for regression, mixing far-apart inputs gives inputs whose true label is not the mix.

## Data
Tabular/time-series (Airfoil, NO2, Exchange-rate, Electricity), EchoNet echocardiogram videos (Echo, ejection fraction), task generalisation (ShapeNet1D, PASCAL3D), OOD (RCF-MNIST, PovertyMap, Crime, SkillCraft, DTI) (verified-from-source).

## Method
Gaussian kernel on label distance with bandwidth sigma defines partner sampling; standard mixup interpolation of inputs and labels; batch variant for large data (verified-from-source).

## Results
- Echo RMSE/MAPE: 5.402/8.700 % (ERM) -> 5.177/8.435 % (C-Mixup); plain mixup worsens MAPE to 8.838 % (verified-from-source).
- Airfoil MAPE 1.753 % -> 1.610 %; Electricity 13.861 % -> 13.372 % (verified-from-source).
- OOD average RMSE RCF-MNIST 0.162 -> 0.146 (verified-from-source).

## Limitations
- Author-stated: not read.
- Reviewer: pixel mixing of cell images creates overlaid mosaics whose CV/HEX is not the label average; feature-level (manifold) mixing would be more defensible for us.

## Relevance to CD / CV / HEX
A label-aware regulariser could reduce over-fitting of large backbones on 8953 images, but the problem we face is plateaued regression with label noise, not lack of augmentation; expected effect is small.

## What it suggests we try
Low priority. If a head is retrained on frozen features, try feature-level C-Mixup (mix embeddings and labels of label-near pairs, sigma about one label SE) and keep it only if OOF MAPE drops.

## Related notes
- [[kim2024confrag]] — C-Mixup is the strongest baseline in the noisy-label regression benchmark.
- [[gong2022ranksim]] — another label-geometry regulariser.

## Sources
- https://ar5iv.labs.arxiv.org/html/2210.05775
- https://neurips.cc/virtual/2022/poster/55153 (via search)
