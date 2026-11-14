---
id: wang2021tent
type: paper
title: "Tent: Fully Test-time Adaptation by Entropy Minimization"
authors: [Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell]
year: 2021
venue: "ICLR 2021"
doi: ""
arxiv: "2006.10726"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/2006.10726"
citation: "Wang D, Shelhamer E, Liu S, Olshausen B, Darrell T. Tent: Fully Test-time Adaptation by Entropy Minimization. ICLR 2021. arXiv:2006.10726"
tier: candidate
group: t7-domain-shift
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html]
evidence: peer-reviewed
tldr: "At test time, re-estimate normalisation statistics on target batches and optimise only the normalisation affine parameters to minimise prediction entropy; ImageNet-C error 44.0 % online vs 49.9 % for test-time BN; needs class probabilities and batches, fails on some shifts."
relevance_clearec: low
relevance_metrics: [CD, CV, HEX]
relevance_note: "Reference TTA baseline; not directly usable for our regression heads (no entropy) unless CV/HEX are cast as binned classification, and its gains rest on target-batch BN statistics that hurt regression in other work."
code_release: open
weights_release: none
data_scale: "ImageNet-C, CIFAR-10/100-C, SVHN->MNIST/MNIST-M/USPS, GTA->Cityscapes, VisDA-C"
key_numbers:
  - "ImageNet-C mean error: test-time BN 49.9 %, Tent 44.0 % online, 42.3 % offline (verified-from-source)"
  - "CIFAR-10-C sev 5 error: source 40.8, BN 17.3, Tent 14.3; CIFAR-100-C: 67.2 / 42.6 / 37.3 (verified-from-source)"
  - "MNIST->SVHN: Tent raises error to 79.8 % vs 71.3 % source-only (verified-from-source)"
  - "Updating the full model never beat the unadapted source model; only normalisation affine parameters (<1 % of weights) are updated (verified-from-source)"
results:
  - {task: "image classification under corruption", dataset: "ImageNet-C", metric: "error", value: "44.0 %", comparator: "test-time BN", comparator_value: "49.9 %", locator: "Sec 4", provenance: "verified-from-source"}
suggests_try: "Do not apply Tent to our regression heads; if a binned (classification-style) CV/HEX head is adopted, entropy TTA becomes possible but should first be tested on the old->new source split, given the documented failure modes."
tags: [test-time-adaptation, entropy-minimisation, batch-norm, source-free]
search_keywords: [Tent, fully test-time adaptation, entropy minimization, test-time batch norm]
uses_dataset: [ImageNet-C, CIFAR-10-C, CIFAR-100-C, SVHN, MNIST, VisDA-C, Cityscapes]
uses_method: [entropy minimisation, normalisation statistics re-estimation, affine-parameter adaptation]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Needs batches; fails on MNIST->SVHN (79.8 % vs 71.3 %); no gain on CIFAR-10.1 / ImageNetV2 natural shifts", locator: "Sec 4-5"}
  - {stated_by: reviewer, note: "Classification only; regression needs other objectives (see adachi2025ssa)", locator: "n/a"}
---

# Tent: Fully Test-time Adaptation by Entropy Minimization

## TL;DR
Tent adapts a trained classifier during testing by replacing normalisation statistics with target-batch statistics and optimising only the normalisation scale/shift parameters to lower prediction entropy. It improves corrupted-image benchmarks over test-time BN alone, but it requires softmax outputs and batches and it fails on some shifts and on natural distribution shifts.

## Citation and link
- Wang D, Shelhamer E, Liu S, Olshausen B, Darrell T. ICLR 2021. arXiv:2006.10726
- https://ar5iv.labs.arxiv.org/html/2006.10726

## Problem and motivation
Adapt to a shifted test distribution without source data and without changing training.

## Data
ImageNet-C, CIFAR-10/100-C, digit shifts, GTA->Cityscapes, VisDA-C.

## Method
Online or offline: per batch, normalise with target statistics, minimise mean entropy w.r.t. normalisation gamma/beta (SGD lr 0.00025, batch 64 on ImageNet).

## Results
- ImageNet-C: BN 49.9 %, Tent 44.0 % online / 42.3 % offline (verified-from-source).
- CIFAR-10-C 40.8 -> 17.3 (BN) -> 14.3 (Tent); CIFAR-100-C 67.2 -> 42.6 -> 37.3 (verified-from-source).
- Failure: MNIST->SVHN 71.3 % -> 79.8 % (verified-from-source).

## Limitations
- Author-stated: batch requirement; failure on large shifts; no gain on natural shifts (CIFAR-10.1, ImageNetV2).
- Reviewer: most of the gain on corruption benchmarks comes from BN re-estimation; in regression, BN-adapt can halve R2 ([[adachi2025ssa]]: 0.605 -> 0.318 on California Housing).

## Relevance to CD / CV / HEX
Background: our two image sources and the HEX OOF-to-test gap (0.40 MAPE) are mild acquisition shifts, closer to "natural shift" where Tent did not help. Our backbones (DINOv3, ViT-L) use LayerNorm, not BN, so the statistics-replacement part does not even apply directly.

## What it suggests we try
Nothing actionable for regression heads; background only.

## Related notes
- [[adachi2025ssa]] — the regression-specific TTA
- [[karani2021tta]] — medical TTA adapting only a normalisation front-end
- [[wang2019tta]] — augmentation-based TTA, no parameter updates

## Sources
- https://ar5iv.labs.arxiv.org/html/2006.10726
- https://arxiv.org/abs/2006.10726
