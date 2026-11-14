---
id: yang2021imbalanced
type: paper
title: "Delving into Deep Imbalanced Regression"
authors: [Yuzhe Yang, Kaiwen Zha, Ying-Cong Chen, Hao Wang, Dina Katabi]
year: 2021
venue: "International Conference on Machine Learning (ICML)"
doi: ""
arxiv: "2102.09554"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/2102.09554"
citation: "Yang Y, Zha K, Chen YC, Wang H, Katabi D. Delving into Deep Imbalanced Regression. ICML 2021. arXiv:2102.09554"
tier: core
group: t6-noisy-regression
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: peer-reviewed
tldr: "Defines deep imbalanced regression and proposes label distribution smoothing (LDS, inverse smoothed-density loss weights) and feature distribution smoothing (FDS); they cut few-shot (tail) error a lot, overall error a little, and sometimes raise many-shot error."
relevance_clearec: high
relevance_metrics: [CV, HEX, CD]
relevance_note: "Our CV errors concentrate in the outer label deciles; DIR is the standard toolkit for tail-range regression, but its objective (balanced error) is not our objective (natural-distribution MAPE)."
code_release: open
weights_release: unknown
data_scale: "IMDB-WIKI-DIR 191.5K train, AgeDB-DIR 12.2K, NYUD2-DIR 50K, STS-B-DIR 5.2K, SHHS-DIR 1,892"
key_numbers:
  - "AgeDB-DIR MAE All/Many/Medium/Few: vanilla 7.77/6.62/9.55/13.67 vs LDS+FDS 7.55/7.01/8.24/10.79 (verified-from-source)"
  - "IMDB-WIKI-DIR MAE All/Many/Medium/Few: vanilla 8.06/7.23/15.12/26.33 vs LDS+FDS 7.78/7.20/12.61/22.19 (verified-from-source)"
  - "LDS default: Gaussian kernel size 5, sigma 2, loss weights = inverse of smoothed label density (verified-from-source)"
  - "Many-shot MAE rises on AgeDB-DIR from 6.62 to 7.01 with LDS+FDS (verified-from-source)"
  - "FDS shows tail-range features inherit dense-range statistics (ages 0-6 look like age 30) (verified-from-source)"
results:
  - {task: "age regression", dataset: "AgeDB-DIR", metric: "MAE all", value: "7.55", comparator: "vanilla", comparator_value: "7.77", locator: "Table (AgeDB-DIR main results)", provenance: "verified-from-source"}
  - {task: "age regression", dataset: "IMDB-WIKI-DIR", metric: "MAE few-shot", value: "22.19", comparator: "vanilla", comparator_value: "26.33", locator: "Table (IMDB-WIKI-DIR main results)", provenance: "verified-from-source"}
suggests_try: "Try LDS-style weights on the CV/HEX loss only with a softened exponent (e.g. inverse sqrt density) and judge purely by overall OOF MAPE; expect tail-decile error down but central error up, so net gain is uncertain (the paper's overall gain is only about 3 % relative)."
tags: [imbalanced-regression, LDS, FDS, tail-error, reweighting]
search_keywords: [deep imbalanced regression, label distribution smoothing, feature distribution smoothing]
uses_dataset: [IMDB-WIKI-DIR, AgeDB-DIR, NYUD2-DIR, STS-B-DIR]
uses_method: [LDS, FDS, loss-reweighting]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Minor degradation in many-shot region (e.g. NYUD2-DIR)", locator: "Results"}
  - {stated_by: reviewer, note: "Optimises balanced error; on natural-distribution test sets gains are small; no label-noise model, so tail labels that are partly noise get up-weighted", locator: "Sec 4-5"}
---

# Delving into Deep Imbalanced Regression

## TL;DR
The paper shows that regression with skewed continuous labels under-serves rare target ranges, and proposes LDS (re-weight the loss by the inverse of a kernel-smoothed label density) and FDS (smooth per-bin feature statistics across neighbouring targets). Tail error falls sharply; overall error improves slightly; the dense middle can get slightly worse.

## Citation and link
- Yang Y, Zha K, Chen YC, Wang H, Katabi D. ICML 2021. arXiv:2102.09554.
- Read: https://ar5iv.labs.arxiv.org/html/2102.09554

## Problem and motivation
Unlike classes, continuous targets have neighbours: nearby targets share information, so empirical label density is a poor measure of how "rare" a target really is. Smoothing in label space gives a better effective density.

## Data
IMDB-WIKI-DIR 191.5K/11.0K/11.0K, AgeDB-DIR 12.2K/2.1K/2.1K, STS-B-DIR 5.2K/1K/1K, NYUD2-DIR 50K train / 654 test, SHHS-DIR 1,892/369/369; validation and test sets are made balanced (verified-from-source).

## Method
- LDS: effective density = label histogram convolved with a symmetric kernel (Gaussian, size 5, sigma 2); loss weight proportional to its inverse (verified-from-source).
- FDS: per-bin feature mean/covariance, smoothed across bins, used to whiten and re-colour features; momentum 0.9 (verified-from-source).
- Combined with other re-weighting, focal-R, RRT (two-stage retraining).

## Results
- AgeDB-DIR MAE All/Many/Medium/Few: 7.77/6.62/9.55/13.67 (vanilla) -> 7.55/7.01/8.24/10.79 (LDS+FDS) (verified-from-source).
- IMDB-WIKI-DIR: 8.06/7.23/15.12/26.33 -> 7.78/7.20/12.61/22.19 (verified-from-source).
- Many-shot MAE rises in several rows (AgeDB 6.62 -> 7.01) (verified-from-source).

## Limitations
- Author-stated: minor many-shot degradation.
- Reviewer: test sets are balanced, which flatters tail methods; under a natural test distribution and per-sample MAPE, the trade-off may be net zero. Tail labels with high noise get up-weighted.

## Relevance to CD / CV / HEX
Our CV predictions span only 66 % of the label spread and the two outer deciles carry about a third of CV error. DIR attacks exactly that, but at a price in the dense middle. Also, part of the outer-decile error is irreducible label noise (about 6.5 % SE on CV), which re-weighting would amplify. FDS's finding that tail features collapse onto dense-range features is a plausible mechanism for our compressed predictions.

## What it suggests we try
A quick ablation: inverse-sqrt smoothed-density weights on the CV head (kernel sigma about 2 label steps), compared on overall OOF MAPE and on the outer-decile share. Adopt only if overall OOF MAPE drops; do not judge by tail error.

## Related notes
- [[gong2022ranksim]] — feature-space ranking regulariser evaluated on the same DIR benchmarks.
- [[yao2022cmixup]] — label-aware mixup, another DIR-benchmark method.
- [[delange2022mindgap]] — regression to the mean in brain-age prediction; label-using corrections inflate accuracy.

## Sources
- https://ar5iv.labs.arxiv.org/html/2102.09554
