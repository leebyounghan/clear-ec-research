---
id: adachi2025ssa
type: paper
title: "Test-time Adaptation for Regression by Subspace Alignment"
authors: [Kazuki Adachi, Shin'ya Yamaguchi, Atsutoshi Kumagai, Tomoki Hamagami]
year: 2025
venue: "ICLR 2025"
doi: ""
arxiv: "2410.03263"
pmcid: ""
url: "https://arxiv.org/html/2410.03263v2"
citation: "Adachi K, Yamaguchi S, Kumagai A, Hamagami T. Test-time Adaptation for Regression by Subspace Alignment. ICLR 2025. arXiv:2410.03263"
tier: core
group: t7-domain-shift
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: peer-reviewed
tldr: "Source-free TTA for regression (SSA): store mean/covariance of penultimate features, project onto the top-K PCA subspace, weight directions by their effect on the regressor, and align target features with a symmetric KL by updating only normalisation-layer affine parameters; classification TTA methods (BN-adapt, prototypes, feature restoration, TTT) often HURT regression."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "The one TTA method designed for regression heads like ours; also a warning that plain BN-statistics adaptation can sharply reduce R2, relevant to our two image sources and the HEX OOF-to-test gap."
code_release: open
weights_release: none
data_scale: "SVHN->MNIST digit regression, UTKFace age with 13 ImageNet-C corruptions, Biwi Kinect head pose (gender shift), California Housing (non-coastal->coastal)"
key_numbers:
  - "UTKFace mean over 13 corruptions R2: source -0.678, BN-adapt 0.654, feature restoration 0.700, SSA 0.731, oracle 0.789 (verified-from-source)"
  - "California Housing R2: source 0.605, BN-adapt 0.318, prototype -0.726, SSA 0.639 (verified-from-source)"
  - "Biwi Kinect R2: source 0.706, BN-adapt 0.746, feature restoration -0.357, TTT 0.288, SSA 0.778 (verified-from-source)"
  - "SVHN->MNIST R2: source 0.406, BN-adapt 0.396, SSA 0.511; naive all-dimension alignment 0.333 (verified-from-source)"
  - "Defaults: K=100 PCA dims of 2048-d features, batch 64, one offline epoch over target data; K>=400 unstable on SVHN-MNIST (verified-from-source)"
results:
  - {task: "age regression under corruption", dataset: "UTKFace-C (mean)", metric: "R2", value: "0.731", comparator: "BN-adapt", comparator_value: "0.654", locator: "Table 4", provenance: "verified-from-source"}
  - {task: "tabular regression shift", dataset: "California Housing", metric: "R2", value: "0.639", comparator: "BN-adapt", comparator_value: "0.318", locator: "Table 3", provenance: "verified-from-source"}
suggests_try: "Before any test-time BN re-estimation on new/darker frames, check on the old-vs-new source split (CD MAPE 7.11 vs 5.50) whether BN-adapt helps or hurts; if TTA is allowed, try SSA (K about 100, affine-only updates) on the regression heads and measure per-source OOF MAPE, especially HEX."
tags: [test-time-adaptation, regression, source-free, feature-alignment, batch-norm]
search_keywords: [SSA, significant-subspace alignment, regression TTA, source-free domain adaptation regression]
uses_dataset: [UTKFace, Biwi Kinect, California Housing, SVHN, MNIST]
uses_method: [PCA subspace, symmetric KL, normalisation-affine adaptation]
compares_with:
  - {target: "[[wang2021tent]]", result: better, metric: "applicability", note: "Tent/entropy methods not evaluated because they need class probabilities", locator: "Sec 2"}
extends: []
has_limitation:
  - {stated_by: author, note: "Assumes covariate shift only (p(y|x) unchanged); diagonal-Gaussian feature assumption may not hold", locator: "Limitations"}
  - {stated_by: author, note: "Sensitive to K (large K diverges); offline epoch over target data with batch 64", locator: "Sec 5, ablation"}
  - {stated_by: reviewer, note: "No medical or microscopy data; label-noise and device-label shift (our device computes CV differently) would violate the covariate-shift assumption", locator: "n/a"}
---

# Test-time Adaptation for Regression by Subspace Alignment

## TL;DR
Most test-time adaptation (TTA) methods rely on class probabilities or align every feature dimension. For regression, features lie in a low-dimensional subspace, so naive alignment adds noise. SSA aligns target features to stored source statistics only in the top PCA directions, weighted by how much each direction moves the regressor's output, updating only normalisation affine parameters. It beats source-only and classification-style TTA on four regression shifts, while several baselines fall below source-only.

## Citation and link
- Adachi K, Yamaguchi S, Kumagai A, Hamagami T. ICLR 2025. arXiv:2410.03263
- https://arxiv.org/html/2410.03263v2 ; code https://github.com/kzkadc/regression-tta

## Problem and motivation
Regression models under covariate shift get no help from entropy-based TTA (no softmax). The authors show full-dimension feature alignment can make things worse (SVHN-MNIST R2 0.406 -> 0.333).

## Data
SVHN->MNIST (digit value as regression target), UTKFace age with 13 ImageNet-C corruptions at max severity, Biwi Kinect head pose with gender shift, California Housing non-coastal -> coastal.

## Method
- Source: mean and covariance of penultimate features (2048-d ResNet), PCA top-K (K=100).
- Target loss: symmetric KL between diagonal Gaussians in the subspace; dimension weight alpha_d = 1 + |w^T v_d| (w = regressor weights).
- Update only gamma/beta of normalisation layers; batch 64; one epoch offline; no source data needed at test time.

## Results
- UTKFace-C mean R2: source -0.678, BN-adapt 0.654, FR 0.700, SSA 0.731 (verified-from-source).
- California Housing R2: source 0.605, BN-adapt 0.318, prototype -0.726, SSA 0.639 (verified-from-source).
- Biwi Kinect R2: source 0.706, BN-adapt 0.746, FR -0.357, TTT 0.288, SSA 0.778 (verified-from-source).
- Table 1/6 list UTKFace source R2 0.020 vs Table 4 mean -0.678, unexplained (reviewer note).

## Limitations
- Author-stated: covariate-shift assumption, Gaussian approximation, K sensitivity.
- Reviewer: gains on mild shifts are small (California 0.605 -> 0.639); needs a batch of target images, i.e. offline TTA on the test set, which the challenge rules may or may not allow; ViT backbones with LayerNorm untested here.

## Relevance to CD / CV / HEX
Our heads are regressors on large pretrained backbones (DINOv3, ViT-L, Cellpose-SAM encoder). We have two image sources (CD error 7.11 old vs 5.50 new; 2023-24 darker, vignetted) and HEX on the hidden test was 0.40 MAPE worse than OOF, a sign of shift. SSA is the regression-appropriate TTA if test-time adaptation is allowed. Its strongest lesson for us is negative: BN-adapt can halve R2 (0.605 -> 0.318), so any batch-statistics trick needs an OOF check first.

## What it suggests we try
Simulate the shift with the two sources (train on old, adapt on new) and compare source-only, BN/LN-stat re-estimation and SSA on CD/CV/HEX MAPE; adopt TTA only if it helps on that proxy and is permitted.

## Related notes
- [[wang2021tent]] — entropy TTA, not applicable to regression heads
- [[karani2021tta]] — per-image adaptation of a normalisation module in medical segmentation
- [[wang2019tta]] — augmentation-based TTA, the safer zero-training alternative

## Sources
- https://arxiv.org/html/2410.03263v2
- https://www.arxiv.org/abs/2410.03263v2
