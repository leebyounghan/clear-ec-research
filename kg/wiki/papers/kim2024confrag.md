---
id: kim2024confrag
type: paper
title: "Sample Selection via Contrastive Fragmentation for Noisy Label Regression"
authors: [Chris Dongjoo Kim, Sangwoo Moon, Jihwan Moon, Dongyeon Woo, Gunhee Kim]
year: 2024
venue: "Advances in Neural Information Processing Systems (NeurIPS)"
doi: ""
arxiv: "2502.17771"
pmcid: ""
url: "https://arxiv.org/html/2502.17771v1"
citation: "Kim CD, Moon S, Moon J, Woo D, Kim G. Sample Selection via Contrastive Fragmentation for Noisy Label Regression. NeurIPS 2024. arXiv:2502.17771"
tier: candidate
group: t6-noisy-regression
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html, website]
evidence: peer-reviewed
tldr: "Noisy-label regression benchmark (age, price, music year) with symmetric and Gaussian label noise; ConFrag selects clean samples by training binary experts on far-apart label fragments and checking neighbourhood agreement, beating 14 baselines including C-Mixup."
relevance_clearec: medium
relevance_metrics: [CD, CV, HEX]
relevance_note: "Shows sample selection helps when a subset of labels is corrupted; our CV/HEX noise is mostly benign sampling noise on every label, but CD has label/marker ratios up to 1.43 that look like corrupted cases."
code_release: unknown
weights_release: unknown
data_scale: "AFAD-B, IMDB-Clean-B, IMDB-WIKI-B, SHIFT15M-B, MSD-B (sizes not read)"
key_numbers:
  - "IMDB-Clean-B, Gaussian noise 30 %/50 %: MRAE ConFrag 15.70/33.36 vs vanilla 26.89/50.28 vs C-Mixup 21.92/47.04 (verified-from-source)"
  - "AFAD-B, Gaussian 30 %/50 %: MRAE ConFrag 17.49/27.31 vs vanilla 28.77/39.03 (verified-from-source)"
  - "Noise settings: symmetric 20-80 %; Gaussian 30 %/50 % with std randomised up to that share of the label range (verified-from-source)"
  - "Default F = 4 label fragments, paired to maximise label distance; one binary expert per pair (verified-from-source)"
results:
  - {task: "age regression with label noise", dataset: "IMDB-Clean-B", metric: "MRAE (Gaussian 30 %)", value: "15.70", comparator: "C-Mixup", comparator_value: "21.92", locator: "Table 1", provenance: "verified-from-source"}
suggests_try: "For CD only: flag training images whose OOF residual is large and consistent across folds/backbones (or whose label/marker-density ratio is far above 1) and down-weight them in the final full-data retrain; do not apply sample selection to CV/HEX, where noise is spread over all labels."
tags: [noisy-labels, regression, sample-selection, benchmark]
search_keywords: [noisy label regression, ConFrag, sample selection regression]
uses_dataset: [AFAD, IMDB-Clean, IMDB-WIKI, SHIFT15M, MSD]
uses_method: [sample-selection, mixture-of-experts]
compares_with:
  - {target: "[[yao2022cmixup]]", result: better, metric: "MRAE", note: "C-Mixup is the strongest non-ConFrag baseline on IMDB-Clean-B", locator: "Table 1"}
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Noise levels (std up to 30-50 % of label range) are far larger than ours; benign homogeneous noise is not the target case", locator: "Sec 5"}
---

# Sample Selection via Contrastive Fragmentation for Noisy Label Regression

## TL;DR
ConFrag splits the label range into fragments, trains binary experts on maximally distant fragment pairs, and keeps samples whose label agrees with the experts' neighbourhood vote. On a new noisy-regression benchmark it beats 14 baselines under symmetric and Gaussian label noise.

## Citation and link
- Kim CD, Moon S, Moon J, Woo D, Kim G. NeurIPS 2024. arXiv:2502.17771.
- Read: https://arxiv.org/html/2502.17771v1 ; NeurIPS poster page (search listing).

## Problem and motivation
Noisy-label methods are mostly built for classification; regression labels are ordered and continuous, so "clean vs noisy" must use label proximity.

## Data
Curated "-B" versions of AFAD, IMDB-Clean, IMDB-WIKI (real web noise), SHIFT15M (price), MSD (music year); UTKFace in appendix (verified-from-source). Sizes not read.

## Method
F = 4 fragments, contrastive pairing, one binary expert per pair, mixture of neighbouring fragments with label-proximity prior, agreement by classifier and k-NN (union), neighbourhood jittering (verified-from-source).

## Results
- IMDB-Clean-B Gaussian 30/50 %: MRAE 15.70/33.36 vs vanilla 26.89/50.28, C-Mixup 21.92/47.04 (verified-from-source).
- AFAD-B Gaussian 30/50 %: 17.49/27.31 vs vanilla 28.77/39.03 (verified-from-source).
- Error Residual Ratio values only in figures (not reported in text).

## Limitations
- Author-stated: not read in detail.
- Reviewer: synthetic noise is large and affects a subset; robust losses (Huber/L1) not compared in the part read; no MAPE analysis.

## Relevance to CD / CV / HEX
CV/HEX labels carry roughly 6.5 %/7.5 % sampling noise on every image, which sample selection cannot remove (it would discard informative images). CD labels are cleaner (about 3 %) but some frames show label/expert-marker density ratios up to 1.43, i.e. a corrupted-subset pattern where selection or down-weighting is appropriate.

## What it suggests we try
Cheap version without ConFrag machinery: in the full-data retrain, down-weight CD samples whose 5-fold OOF absolute percentage error exceeds a high quantile in all backbones; check that OOF CD MAPE on the remaining samples improves and that the flagged set is enriched for one image source (old vs new IDs).

## Related notes
- [[yao2022cmixup]] — strongest baseline here.
- [[yang2021imbalanced]] — tail-range regression on overlapping benchmarks.

## Sources
- https://arxiv.org/html/2502.17771v1
- https://neurips.cc/virtual/2024/poster/95898 (via search)
