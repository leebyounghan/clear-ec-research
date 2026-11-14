---
id: liang2022transcrowd
type: paper
title: "TransCrowd: weakly-supervised crowd counting with transformers"
authors: [Dingkang Liang, Xiwu Chen, Wei Xu, Yu Zhou, Xiang Bai]
year: 2022
venue: "Science China Information Sciences"
doi: "10.1007/s11432-021-3445-y"
arxiv: "2104.09116"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/2104.09116"
citation: "Liang D, Chen X, Xu W, Zhou Y, Bai X. TransCrowd: weakly-supervised crowd counting with transformers. Sci China Inf Sci. 2022;65(6). doi:10.1007/s11432-021-3445-y"
tier: candidate
group: t5-weak-supervision
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html, website]
evidence: peer-reviewed
tldr: "A pure ViT regressing the image-level count (L1, count labels only) beats earlier count-supervised methods and some point-supervised CNNs, but stays clearly behind the best point-supervised density method (DM-Count)."
relevance_clearec: medium
relevance_metrics: [CD]
relevance_note: "Quantifies the gap between count-only regression (our current black-box CD setup) and location-supervised density counting, which our Topcon point set could provide."
code_release: open
weights_release: unknown
data_scale: "ShanghaiTech A/B, UCF-QNRF, NWPU-Crowd"
key_numbers:
  - "ShanghaiTech B MAE/MSE: TransCrowd-GAP (count labels only) 9.3/16.1 vs DM-Count (points) 7.4/11.8 (verified-from-source)"
  - "UCF-QNRF MAE/MSE: TransCrowd-GAP 97.2/168.5 vs DM-Count 85.6/148.3 (verified-from-source)"
  - "ShanghaiTech A MAE: TransCrowd-GAP 66.1 vs prior weakly supervised MATT 80.1 and Yang et al. 104.6 (verified-from-source)"
  - "GAP over tokens beats a learnable regression token on every dataset (verified-from-source)"
  - "Input 384x384, 16x16 patches (576 tokens), 12 layers, ImageNet init, L1 count loss (verified-from-source)"
results:
  - {task: "crowd counting", dataset: "UCF-QNRF", metric: "MAE", value: "97.2 (count-only)", comparator: "DM-Count (point-supervised)", comparator_value: "85.6", locator: "Tables 1-3", provenance: "verified-from-source"}
suggests_try: "Evidence that location supervision beats count-only regression by roughly 12-20 % relative MAE in crowds; for CD, test a point/density-derived CD (Topcon-trained) as an extra input feature or ensemble member to the count-only regressor and check OOF CD MAPE."
tags: [weak-supervision, count-level-labels, transformer, crowd-counting]
search_keywords: [weakly supervised counting count-level labels, TransCrowd, sequence-to-count]
uses_dataset: [ShanghaiTech, UCF-QNRF, NWPU-Crowd]
uses_method: [ViT-regression, global-average-pooling, L1-count-loss]
compares_with:
  - {target: "[[wang2020dmcount]]", result: worse, metric: "MAE", note: "count-only ViT trails point-supervised DM-Count on SHA, SHB, UCF-QNRF, NWPU", locator: "Tables 1-4"}
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Crowd counts span orders of magnitude; in our narrow CD range the relative advantage of localisation may differ", locator: "Experiments"}
---

# TransCrowd: weakly-supervised crowd counting with transformers

## TL;DR
TransCrowd casts counting as sequence-to-count: a ViT reads image patches and regresses the total count from count-level labels only. It is the best count-supervised method of its time but remains behind point-supervised density approaches.

## Citation and link
- Liang D, Chen X, Xu W, Zhou Y, Bai X. Sci China Inf Sci. 2022;65(6). doi:10.1007/s11432-021-3445-y.
- Read: https://ar5iv.labs.arxiv.org/html/2104.09116

## Problem and motivation
Point annotation of crowds is costly; count labels are cheaper. Global self-attention may compensate for missing locations.

## Data
ShanghaiTech A/B, UCF-QNRF, NWPU-Crowd (verified-from-source).

## Method
ViT encoder (384x384, 576 tokens, 12 layers), two heads: regression token or GAP over tokens; L1 loss on count (verified-from-source).

## Results
- Weakly supervised: TransCrowd-GAP SHA 66.1/105.1, SHB 9.3/16.1, UCF-QNRF 97.2/168.5 MAE/MSE (verified-from-source).
- Point-supervised DM-Count: SHA 59.7/95.7, SHB 7.4/11.8, UCF-QNRF 85.6/148.3 (verified-from-source).
- NWPU test: TransCrowd-GAP 117.7/451.0 vs DM-Count 88.4/388.6 (verified-from-source).

## Limitations
- Author-stated: not read.
- Reviewer: crowd domain; no microscopy.

## Relevance to CD / CV / HEX
Our CD model is a count-only (image-level) regressor on big pretrained backbones, the TransCrowd setting. The consistent gap to point-supervised density methods suggests that adding location supervision (Topcon points, our detector) could still help CD, our second-largest gap to first place (about 0.09 MAPE points).

## What it suggests we try
Add the density/point-derived CD (counted in the recorded analysis box) as a feature to the CD head or as an ensemble member with an OOF-fitted weight; keep it if OOF CD MAPE drops below 6.19.

## Related notes
- [[wang2020dmcount]] — point-supervised comparator.
- [[lempitsky2010counting]] — density counting origin.

## Sources
- https://ar5iv.labs.arxiv.org/html/2104.09116
- https://link.springer.com/article/10.1007/s11432-021-3445-y (via search)
