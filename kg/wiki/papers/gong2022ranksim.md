---
id: gong2022ranksim
type: paper
title: "RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression"
authors: [Yu Gong, Greg Mori, Frederick Tung]
year: 2022
venue: "International Conference on Machine Learning (ICML), PMLR 162"
doi: ""
arxiv: "2205.15236"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/2205.15236"
citation: "Gong Y, Mori G, Tung F. RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression. ICML 2022, PMLR 162:7634-7649. arXiv:2205.15236"
tier: candidate
group: t6-noisy-regression
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html, website]
evidence: peer-reviewed
tldr: "Regulariser that makes the ranking of a sample's neighbours in feature space match their ranking in label space (differentiable via blackbox-solver gradients); complementary to LDS/FDS and re-weighting on DIR benchmarks."
relevance_clearec: medium
relevance_metrics: [CV, HEX]
relevance_note: "Targets the feature collapse that produces compressed predictions in rare label ranges, which matches our CV outer-decile error."
code_release: open
weights_release: open
data_scale: "IMDB-WIKI-DIR, AgeDB-DIR, STS-B-DIR"
key_numbers:
  - "AgeDB-DIR MAE All/Many/Med/Few: vanilla 7.77/6.62/9.55/13.67 vs LDS+FDS 7.55/7.01/8.24/10.79 vs LDS+FDS+RankSim 7.03/6.54/7.68/9.92 (verified-from-source)"
  - "IMDB-WIKI-DIR MAE All: vanilla 8.06 vs LDS+FDS+RankSim 7.69; best 7.35 with RRT+FDS+RankSim (verified-from-source)"
  - "Gradient of the ranking operation via blackbox combinatorial solver (Vlastelica et al. 2020); hyperparameters gamma (weight) and lambda (interpolation) (verified-from-source)"
results:
  - {task: "age regression", dataset: "AgeDB-DIR", metric: "MAE all", value: "7.03", comparator: "LDS+FDS", comparator_value: "7.55", locator: "main table", provenance: "verified-from-source"}
suggests_try: "On a frozen-feature head is not applicable (needs backbone gradients); if one more fine-tune runs, add RankSim on the CV label (batch-wise rank loss) and compare overall OOF CV MAPE and prediction spread vs label spread."
tags: [imbalanced-regression, ranking-loss, representation]
search_keywords: [RankSim, ranking similarity regularization, deep imbalanced regression]
uses_dataset: [IMDB-WIKI-DIR, AgeDB-DIR, STS-B-DIR]
uses_method: [rank-regularisation, blackbox-solver-gradients]
compares_with:
  - {target: "[[yang2021imbalanced]]", result: better, metric: "MAE all (AgeDB-DIR)", note: "7.03 with RankSim added vs 7.55 for LDS+FDS", locator: "main table"}
extends: ["[[yang2021imbalanced]]"]
has_limitation:
  - {stated_by: reviewer, note: "Balanced test sets; with noisy labels, enforcing label-rank order in features can fit noise", locator: "Experiments"}
---

# RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression

## TL;DR
RankSim adds a loss that aligns the order of feature-space neighbours with the order of label-space neighbours, using non-local neighbours too. Added to LDS/FDS or re-weighting, it improves all, many, medium and few-shot MAE on standard DIR benchmarks.

## Citation and link
- Gong Y, Mori G, Tung F. ICML 2022, PMLR 162:7634-7649. arXiv:2205.15236.
- Read: https://ar5iv.labs.arxiv.org/html/2205.15236

## Problem and motivation
Smoothing methods (LDS/FDS) use only local label neighbourhoods; RankSim enforces a global label-feature ordering so rare targets are not mapped onto dense-range features.

## Data
IMDB-WIKI-DIR, AgeDB-DIR, STS-B-DIR (from [[yang2021imbalanced]]) (verified-from-source).

## Method
Per batch, rank label similarities and feature similarities for each anchor; penalise rank mismatch; gradient via blackbox combinatorial solver; weight gamma, interpolation lambda (verified-from-source).

## Results
- AgeDB-DIR All/Many/Med/Few: 7.03/6.54/7.68/9.92 with LDS+FDS+RankSim vs 7.55/7.01/8.24/10.79 (verified-from-source).
- IMDB-WIKI-DIR All: 7.69 (LDS+FDS+RankSim), best 7.35 (RRT+FDS+RankSim) (verified-from-source).
- Unlike LDS+FDS alone, many-shot error also improves on AgeDB-DIR (6.54 vs 6.62 vanilla) (verified-from-source).

## Limitations
- Author-stated: not read.
- Reviewer: needs backbone fine-tuning; noise-free age labels.

## Relevance to CD / CV / HEX
Our compressed CV predictions could partly reflect feature collapse at the extremes (as FDS shows for ages); RankSim is the regulariser that improved many-shot and few-shot at the same time on AgeDB-DIR, unlike pure re-weighting.

## What it suggests we try
Only if an end-to-end fine-tune is planned anyway: add RankSim on CV labels; otherwise skip (frozen features cannot benefit).

## Related notes
- [[yang2021imbalanced]] — base DIR benchmark and LDS/FDS.
- [[yao2022cmixup]] — label-aware mixup alternative.

## Sources
- https://ar5iv.labs.arxiv.org/html/2205.15236
- https://proceedings.mlr.press/v162/gong22a.html (via search)
