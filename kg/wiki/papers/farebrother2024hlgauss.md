---
id: farebrother2024hlgauss
type: paper
title: "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL"
authors: [Jesse Farebrother, Jordi Orbay, Quan Vuong, Adrien Ali Taïga, Yevgen Chebotar, Ted Xiao, et al.]
year: 2024
venue: "International Conference on Machine Learning (ICML), PMLR 235"
doi: ""
arxiv: "2403.03950"
pmcid: ""
url: "https://arxiv.org/html/2403.03950"
citation: "Farebrother J, Orbay J, Vuong Q, Taïga AA, Chebotar Y, Xiao T, et al. Stop Regressing: Training Value Functions via Classification for Scalable Deep RL. Proceedings of ICML 2024, PMLR 235:13049-13071. arXiv:2403.03950"
tier: core
group: t6-noisy-regression
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, website]
evidence: peer-reviewed
tldr: "Replacing MSE regression with cross-entropy on a histogram whose target is a Gaussian-smoothed bin distribution (HL-Gauss, sigma/bin = 0.75) consistently improves deep value-function learning and degrades more gracefully than MSE under noisy targets."
relevance_clearec: high
relevance_metrics: [CV, HEX, CD]
relevance_note: "Our CV labels are already discrete (step 0.01, 78 values) and noisy; an HL-Gauss head with sigma set to the known label noise gives a full predictive distribution we can decode for MAPE."
code_release: partial
weights_release: none
data_scale: "Atari 2600 (single and multi-task), robotic manipulation, chess, Wordle (RL)"
key_numbers:
  - "HL-Gauss: target = Gaussian (mean = scalar target, std sigma) integrated over each bin; default sigma/bin-width = 0.75, spreading mass to about 6 bins; decode = expectation (verified-from-source)"
  - "About 30 % IQM improvement with SoftMoE scaling on single-task Atari (verified-from-source)"
  - "With reward noise eta in {0.1, 0.3, 1.0}, HL-Gauss degrades more gracefully than MSE; no numbers given in text (verified-from-source)"
  - "HL-Gauss substantially outperforms Two-Hot and MSE losses on online Atari (verified-from-source)"
  - "Gain over MSE largely disappears in offline SARSA, so part of the benefit is handling non-stationary targets (verified-from-source)"
results:
  - {task: "value learning", dataset: "single-task Atari with SoftMoE", metric: "IQM", value: "about +30 %", comparator: "MSE regression", comparator_value: "baseline", locator: "Introduction / results", provenance: "verified-from-source"}
suggests_try: "Swap the CV (and HEX) regression head for a histogram head: 78+ bins at step 0.01 for CV, HL-Gauss targets with sigma about 2-3 bins (close to the 6.5 % label SE at CV 0.38), trained with cross-entropy; decode the 1/y-weighted median per image rather than the expectation, and compare OOF MAPE against the current head."
tags: [regression-as-classification, histogram-loss, HL-Gauss, label-smoothing, noisy-targets]
search_keywords: [HL-Gauss, histogram loss, two-hot, cross-entropy regression]
uses_dataset: [Atari 2600]
uses_method: [HL-Gauss, histogram-loss, cross-entropy]
compares_with:
  - {target: "[[stewart2023regclass]]", result: similar, metric: "regression vs classification", note: "both find classification heads beat MSE; Farebrother adds Gaussian smoothing and noise robustness", locator: "Related work"}
extends: []
has_limitation:
  - {stated_by: author, note: "Benefit partly tied to non-stationary RL targets; smaller under stationary (offline SARSA) targets", locator: "Analysis section"}
  - {stated_by: reviewer, note: "No supervised image-regression benchmark; transfer of the gain to stationary supervised targets is not shown", locator: "Experiments"}
---

# Stop Regressing: Training Value Functions via Classification for Scalable Deep RL

## TL;DR
Value networks trained with categorical cross-entropy on a Gaussian-smoothed histogram target (HL-Gauss) beat MSE regression across Atari, robotics, chess and language agents, and are more robust to noisy targets. The paper popularised a simple recipe: bins, Gaussian label smoothing with sigma/bin = 0.75, cross-entropy, expectation decoding.

## Citation and link
- Farebrother J, et al. ICML 2024, PMLR 235:13049-13071. arXiv:2403.03950.
- Read: https://arxiv.org/html/2403.03950 ; https://proceedings.mlr.press/v235/farebrother24a.html (search listing).

## Problem and motivation
Deep RL value functions scale poorly with network size under MSE. Classification losses scale well in vision and language; the authors test whether casting value regression as classification helps.

## Data
RL benchmarks: single-task and multi-task Atari 2600, robotic manipulation with Q-transformers, chess without search, Wordle (verified-from-source).

## Method
- Bins of width varsigma over the target range; target distribution = Gaussian(mean = target, std sigma) integrated per bin (HL-Gauss, from Imani and White 2018); cross-entropy loss; prediction = expectation over bins.
- Compared with Two-Hot (mass on two neighbouring bins), C51-style categorical, MSE.

## Results
- About 30 % IQM gain with SoftMoE on single-task Atari; 1.8-2.1x on multi-task Atari; other domain gains listed in the introduction (verified-from-source).
- HL-Gauss substantially outperforms Two-Hot and MSE on online Atari (verified-from-source).
- Under reward noise, HL-Gauss degrades more gracefully than MSE (no numbers in text) (verified-from-source).

## Limitations
- Author-stated: part of the gain comes from non-stationarity; smaller under offline SARSA.
- Reviewer: RL-only evidence; for supervised regression with stationary labels the gain may be smaller.

## Relevance to CD / CV / HEX
CV labels have 78 distinct values at step 0.01; HEX is also coarsely quantised. A histogram head matches that structure, Gaussian smoothing with sigma equal to the known sampling noise (about 6.5 % on CV, 7.5 % on HEX) encodes our label-noise model, and the output distribution lets us decode the MAPE-optimal functional ([[gneiting2011point]], [[demyttenaere2016mape]]) instead of the mean.

## What it suggests we try
On the frozen-backbone features (cheap, < 1 h): fit an HL-Gauss head for CV with sigma in {1, 2, 3} bins; report OOF MAPE for expectation, median and 1/y-weighted-median decoding vs the current regression head. If it wins, repeat for HEX, then CD (CD bins in log space).

## Related notes
- [[stewart2023regclass]] — theory for why classification heads learn different features.
- [[gneiting2011point]] — which functional of the histogram to decode under MAPE.
- Cao et al. 2020 (CORAL, rank-consistent ordinal regression) — ordinal alternative for discrete labels; no note in this KG.

## Sources
- https://arxiv.org/html/2403.03950
- https://proceedings.mlr.press/v235/farebrother24a.html (via search)
