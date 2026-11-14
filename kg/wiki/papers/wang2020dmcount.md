---
id: wang2020dmcount
type: paper
title: "Distribution Matching for Crowd Counting"
authors: [Boyu Wang, Huidong Liu, Dimitris Samaras, Minh Hoai]
year: 2020
venue: "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)"
doi: ""
arxiv: "2009.13077"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/2009.13077"
citation: "Wang B, Liu H, Samaras D, Hoai M. Distribution Matching for Crowd Counting. Advances in Neural Information Processing Systems 33 (NeurIPS 2020). arXiv:2009.13077"
tier: candidate
group: t5-weak-supervision
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html, website]
evidence: peer-reviewed
tldr: "Counting loss = count L1 + Sinkhorn optimal-transport distance between normalised predicted and dot maps + total variation; avoids Gaussian smoothing of dots, which the authors prove loosens the generalisation bound, and lowers counting MAE on all major crowd benchmarks."
relevance_clearec: medium
relevance_metrics: [CD]
relevance_note: "A drop-in loss for training a cell-density head directly on Topcon dot annotations, with an explicit image-level count term that can take the CD-derived count."
code_release: open
weights_release: open
data_scale: "UCF-QNRF, NWPU-Crowd, ShanghaiTech A/B, UCF-CC-50"
key_numbers:
  - "Loss = |predicted count - true count| + OT (Sinkhorn, 100 iterations) between normalised maps + TV distance (verified-from-source)"
  - "UCF-QNRF MAE/RMSE: DM-Count 85.6/148.3 vs Bayesian loss 88.7/154.8 vs Gaussian pixel-wise loss 106.8/183.7 (verified-from-source)"
  - "ShanghaiTech B MAE: 7.4 vs Bayesian 7.7 vs Gaussian pixel-wise 8.5 (verified-from-source)"
  - "NWPU test MAE 88.4 vs 105.4 (Bayesian); NAE 0.169 vs 0.203 (verified-from-source)"
  - "Theorem: generalisation bound of Gaussian-smoothed training depends on the smoothed-vs-real annotation gap; DM-Count bounds are tighter (verified-from-source)"
results:
  - {task: "crowd counting", dataset: "UCF-QNRF", metric: "MAE", value: "85.6", comparator: "Gaussian-smoothed pixel-wise L2", comparator_value: "106.8", locator: "main results table", provenance: "verified-from-source"}
suggests_try: "If a density head is trained on the 385 Topcon point images, use DM-Count's OT + count loss instead of Gaussian-smoothed L2; on CLEAR-EC frames without points, keep only the count term with the CD-derived count in the analysis box (a weak, image-level signal)."
tags: [counting, optimal-transport, Sinkhorn, density-map, loss-design]
search_keywords: [DM-Count, distribution matching counting, optimal transport counting loss]
uses_dataset: [UCF-QNRF, NWPU-Crowd, ShanghaiTech, UCF-CC-50]
uses_method: [optimal-transport-loss, Sinkhorn, total-variation-loss, count-loss]
compares_with: []
extends: ["[[lempitsky2010counting]]"]
has_limitation:
  - {stated_by: reviewer, note: "Crowd scenes with large count range; benefit for dense, regular cell mosaics with near-constant cell size is untested", locator: "Experiments"}
---

# Distribution Matching for Crowd Counting

## TL;DR
DM-Count trains a density network by matching the normalised predicted map to the dot map with optimal transport (Sinkhorn), plus a count term and a TV term. It shows Gaussian smoothing of dots harms generalisation and improves MAE over the Bayesian loss and pixel-wise L2 on five benchmarks.

## Citation and link
- Wang B, Liu H, Samaras D, Hoai M. NeurIPS 2020. arXiv:2009.13077.
- Read: https://ar5iv.labs.arxiv.org/html/2009.13077 ; NeurIPS proceedings listing (via search).

## Problem and motivation
Density targets made by Gaussian-blurring dots depend on a hand-picked kernel; the Bayesian loss is underdetermined. OT compares spatial distributions directly.

## Data
UCF-QNRF, NWPU-Crowd, ShanghaiTech A/B, UCF-CC-50 (verified-from-source).

## Method
Counting loss (L1 on totals), OT loss on normalised maps via Sinkhorn (100 iterations), TV loss for stability; VGG-19 backbone (verified-from-source for losses).

## Results
- UCF-QNRF MAE/RMSE 85.6/148.3 vs Bayesian 88.7/154.8 vs pixel-wise 106.8/183.7 (verified-from-source).
- ShanghaiTech A 59.7/95.7; B 7.4/11.8; UCF-CC-50 211.0/291.5 (verified-from-source).
- NWPU test MAE 88.4, NAE 0.169 (verified-from-source).

## Limitations
- Author-stated: not read.
- Reviewer: evidence for the smoothing argument is theoretical plus qualitative.

## Relevance to CD / CV / HEX
For CD, a density head integrates to a count over the recorded analysis box. The count term of DM-Count accepts an image-level number, so CLEAR-EC frames (no points) can contribute through the CD-derived count while Topcon frames supply full OT supervision.

## What it suggests we try
Low priority versus regression fixes; if a density head is trained anyway, use OT + count loss and evaluate CD MAPE of density-integral CD on OOF.

## Related notes
- [[lempitsky2010counting]] — density-map counting origin.
- [[liang2022transcrowd]] — count-only supervision.

## Sources
- https://ar5iv.labs.arxiv.org/html/2009.13077
- https://papers.nips.cc/paper/2020/hash/118bd558033a1016fcc82560c65cca5f-Abstract.html (via search)
