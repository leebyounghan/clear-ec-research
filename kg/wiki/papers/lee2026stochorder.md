---
id: lee2026stochorder
type: paper
title: "Stochastic Order Learning: An Approach to Rank Estimation Using Noisy Data"
authors: [Chaewon Lee, Seon-Ho Lee, Chang-Su Kim]
year: 2026
venue: "International Conference on Machine Learning (ICML) 2026 (accepted; arXiv)"
doi: ""
arxiv: "2607.08103"
pmcid: ""
url: "https://arxiv.org/html/2607.08103v1"
citation: "Lee C, Lee SH, Kim CS. Stochastic Order Learning: An Approach to Rank Estimation Using Noisy Data. ICML 2026 (accepted). arXiv:2607.08103"
tier: core
group: t6-noisy-regression
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: preprint
tldr: "Treats each noisy ordinal label as a discrete-Gaussian distribution over plausible true ranks and trains rank centroids with a discriminative loss plus a probabilistic pairwise order loss; best or near-best MAE on noisy age, aesthetics, bone-age and MT-score benchmarks, ahead of noisy-regression methods (ConFrag, C-Mixup)."
relevance_clearec: high
relevance_metrics: [CV, HEX]
relevance_note: "Most recent method built for exactly our label type: discrete ordered targets with zero-mean noise of known shape (CV step 0.01, about 6.5 % sampling noise); also a medical regression benchmark (RSNA bone age)."
code_release: open
weights_release: unknown
data_scale: "MORPH II, CLAP2015 (age), AADB (aesthetics), RSNA (bone age), WMT2020 (MT quality scores)"
key_numbers:
  - "Noise model: zero-mean discrete Gaussian over neighbouring ranks; synthetic levels kappa 0.2-0.4 (age), 0.1-0.2 (RSNA); Laplacian, uniform, skewed variants also tested (verified-from-source)"
  - "CLAP2015, Gaussian kappa 0.4, MAE/CS: SOL 4.002/73.68 vs GOL 4.105/72.10 vs ConFrag 5.328/58.20 vs C-Mixup 5.302/58.52 (verified-from-source)"
  - "MORPH II, kappa 0.4: SOL 2.826 vs MWR 2.851 vs GOL 2.861 MAE (verified-from-source)"
  - "RSNA bone age, kappa 0.1: SOL 7.579 vs MWR 7.833 vs GOL 8.170 MAE (verified-from-source)"
  - "Ablation CLAP2015 kappa 0.3: order loss alone 3.785, discriminative loss alone 16.433, full 3.764 MAE (verified-from-source)"
  - "Real noise WMT2020: PCC 0.680 vs 0.660 for best prior (verified-from-source)"
results:
  - {task: "age estimation with label noise", dataset: "CLAP2015", metric: "MAE (kappa 0.4)", value: "4.002", comparator: "ConFrag", comparator_value: "5.328", locator: "main results table", provenance: "verified-from-source"}
  - {task: "bone age with label noise", dataset: "RSNA", metric: "MAE (kappa 0.1)", value: "7.579", comparator: "MWR", comparator_value: "7.833", locator: "main results table", provenance: "verified-from-source"}
suggests_try: "The cheap transferable part is the pairwise order loss weighted by the probability that label i truly exceeds label j under the known noise (Gaussian with SE about 6.5 % on CV): add it to the CV head on frozen features (pairs within a batch) so near-tied pairs carry no gradient while far-apart pairs enforce order; judge by OOF MAPE and prediction spread."
tags: [ordinal-regression, noisy-labels, order-learning, rank-estimation]
search_keywords: [stochastic order learning, noisy rank estimation, ordinal labels noise]
uses_dataset: [MORPH II, CLAP2015, AADB, RSNA bone age, WMT2020]
uses_method: [order-learning, centroid-classification, probabilistic-pairwise-margin-loss]
compares_with:
  - {target: "[[kim2024confrag]]", result: better, metric: "MAE (CLAP2015, kappa 0.4)", note: "4.002 vs 5.328", locator: "main table"}
  - {target: "[[yao2022cmixup]]", result: better, metric: "MAE (CLAP2015, kappa 0.4)", note: "4.002 vs 5.302", locator: "main table"}
extends: []
has_limitation:
  - {stated_by: author, note: "Discrete Gaussian error model chosen for tractability; not best everywhere (RSNA under Laplacian noise)", locator: "Method; results"}
  - {stated_by: reviewer, note: "Preprint (ICML 2026 acceptance stated on arXiv); no clean-label results in the part read; centroid decoding gives ranks, not a MAPE-optimal point", locator: "Experiments"}
---

# Stochastic Order Learning: An Approach to Rank Estimation Using Noisy Data

## TL;DR
SOL models a noisy ordinal label as a probability distribution over nearby true ranks and learns rank centroids with (i) a discriminative instance-centroid loss and (ii) a pairwise order loss weighted by the probabilities that one instance is greater than, equal to or less than another. It is robust across noise shapes and beats both ordinal and noisy-regression baselines.

## Citation and link
- Lee C, Lee SH, Kim CS. ICML 2026 (accepted per arXiv). arXiv:2607.08103 (July 2026).
- Read: https://arxiv.org/html/2607.08103v1 ; https://arxiv.org/abs/2607.08103

## Problem and motivation
Rank labels (age, scores, grades) are noisy; a hard label is better seen as one draw from several plausible ranks, so pairwise orders become probabilistic rather than certain.

## Data
MORPH II and CLAP2015 (age), AADB (aesthetics), RSNA (bone age), WMT2020 direct-assessment scores with real human noise (verified-from-source).

## Method
Discrete Gaussian error model gives weights over neighbouring ranks; discriminative loss to own and nearby centroids; stochastic order loss = margin losses on pairs weighted by P(>), P(=), P(<); closed-form centroid update per epoch; test-time rank = nearest centroid in stochastic dissimilarity; optional outlier relabelling (verified-from-source).

## Results
- CLAP2015 kappa 0.4 MAE/CS: 4.002/73.68 vs GOL 4.105/72.10, ConFrag 5.328/58.20, C-Mixup 5.302/58.52 (verified-from-source).
- MORPH II kappa 0.4: 2.826 vs MWR 2.851 (verified-from-source).
- RSNA kappa 0.1: 7.579 vs MWR 7.833 (verified-from-source).
- AADB kappa 0.4: 0.115 vs ConR 0.118 (verified-from-source).
- Ablation: order loss carries most of the gain (3.785 alone vs 3.764 full) (verified-from-source).

## Limitations
- Author-stated: Gaussian error model for tractability; exceptions under Laplacian noise.
- Reviewer: very recent preprint; gains over strong ordinal baselines are small (MORPH 2.826 vs 2.851); decoding not designed for MAPE.

## Relevance to CD / CV / HEX
CV labels are ordinal with 78 values at step 0.01 and roughly Gaussian sampling noise of known size; HEX is similar. SOL's finding that a noise-aware pairwise order loss beats noisy-regression methods (ConFrag, C-Mixup) by a wide margin on CLAP2015 suggests that for our label type the useful prior is the noise shape, not sample selection. It addresses part (b) (label noise), not the device transform.

## What it suggests we try
Add a probabilistic pairwise order loss on the CV head: for pairs (i, j) in a batch, weight a margin loss by P(true_i > true_j) computed from the two labels and the 6.5 % noise; compare OOF CV MAPE and the prediction/label spread ratio with and without it. Low cost on frozen features.

## Related notes
- [[gao2017dldl]] — same Gaussian-over-neighbours label view, pointwise rather than pairwise.
- [[kim2024confrag]] — noisy-regression baseline it beats.
- [[gong2022ranksim]] — another rank-alignment regulariser.

## Sources
- https://arxiv.org/html/2607.08103v1
- https://arxiv.org/abs/2607.08103
