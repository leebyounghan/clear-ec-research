---
id: karani2021tta
type: paper
title: "Test-time adaptable neural networks for robust medical image segmentation"
authors: [Neerav Karani, Ertunc Erdil, Krishna Chaitanya, Ender Konukoglu]
year: 2021
venue: "Medical Image Analysis"
doi: ""
arxiv: "2004.04668"
pmcid: ""
url: "https://ar5iv.arxiv.org/html/2004.04668"
citation: "Karani N, Erdil E, Chaitanya K, Konukoglu E. Test-time adaptable neural networks for robust medical image segmentation. Med Image Anal. 2021;68:101907. arXiv:2004.04668"
tier: candidate
group: t7-domain-shift
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html]
evidence: peer-reviewed
tldr: "Splits a segmenter into a small 3-layer image-normalisation CNN plus a frozen segmentation CNN; at test time only the normaliser is optimised per image so that the predicted mask agrees with a denoising-autoencoder shape prior; large gains on scanner/protocol shifts in brain and prostate MRI, none on heart, about 12 min per image."
relevance_clearec: low
relevance_metrics: [CD, CV, HEX]
relevance_note: "Shows that adapting only a tiny input-normalisation module (not the backbone) can fix acquisition shift, analogous to our brightness/vignetting difference between image sources; per-image cost is too high for us as published."
code_release: open
weights_release: unknown
data_scale: "Brain (HCP-T1 -> ABIDE-Caltech, HCP-T2), prostate (NCI -> USZ, PROMISE12), heart (ACDC -> RVSC); 104 target test volumes"
key_numbers:
  - "Brain HCP-T2 Dice: source only 0.107, + augmentation 0.083, + test-time adaptation 0.733, per-domain upper bound 0.867 (verified-from-source)"
  - "Brain ABIDE-Caltech Dice: 0.588 / 0.753 / 0.800 / 0.896 (verified-from-source)"
  - "Heart RVSC Dice: 0.670 / 0.744 / 0.742 (no gain over augmentation) / 0.806 (verified-from-source)"
  - "Adapting the whole segmentation network instead lowered brain TD1 Dice to 0.671 (verified-from-source)"
  - "About 1 h for the first image of a new domain, then about 12 min per image (verified-from-source)"
results:
  - {task: "brain segmentation, T1->T2 shift", dataset: "HCP-T2", metric: "Dice", value: "0.733", comparator: "source + augmentation", comparator_value: "0.083", locator: "Table 2", provenance: "verified-from-source"}
suggests_try: "Put a tiny learnable input-normalisation layer (few conv layers or a per-image gain/flat-field field) in front of the frozen backbone and train it with source-specific statistics, so the darker vignetted 2023-24 frames are mapped toward the other source; check per-source CD MAPE."
tags: [test-time-adaptation, medical-segmentation, normalisation-module, shape-prior, scanner-shift]
search_keywords: [test-time adaptable networks, denoising autoencoder prior, image normalisation CNN, MRI domain shift]
uses_dataset: [HCP, ABIDE, NCI-ISBI prostate, PROMISE12, ACDC, RVSC]
uses_method: [normalisation CNN, denoising autoencoder, per-image optimisation]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "No convergence guarantee; DAE input may be out of distribution under large shifts; runtime about 12 min/image", locator: "Discussion"}
  - {stated_by: reviewer, note: "Segmentation only; the shape prior is learned from labels, which we do not have at mask level", locator: "n/a"}
---

# Test-time adaptable neural networks for robust medical image segmentation

## TL;DR
A segmentation network is split into a shallow normalisation CNN and a deep segmentation CNN, both trained on one scanner. For each test image, only the normaliser is optimised so that the output segmentation matches what a denoising autoencoder (trained on source labels) considers plausible. This recovers much of the drop from scanner/protocol changes in brain and prostate MRI.

## Citation and link
- Karani N, Erdil E, Chaitanya K, Konukoglu E. Med Image Anal. 2021;68:101907. arXiv:2004.04668
- https://ar5iv.arxiv.org/html/2004.04668

## Problem and motivation
Models trained on one scanner/protocol fail on others; augmentation alone does not close the gap.

## Data
Brain HCP-T1 source, ABIDE-Caltech-T1 and HCP-T2 targets (FreeSurfer labels); prostate NCI -> USZ and PROMISE12; heart ACDC -> RVSC. 104 target test volumes.

## Method
Normaliser: 3 residual conv layers (16, 16, 1 channels), Gaussian-like activations. Segmenter frozen. Per test image: 500 (brain) or 7,500 (prostate, heart) gradient steps on the normaliser with a Dice loss against DAE-denoised predictions, refreshed every 25 steps.

## Results
- Dice source / +DA / +TTA / upper bound: brain TD1 0.588 / 0.753 / 0.800 / 0.896; brain TD2 0.107 / 0.083 / 0.733 / 0.867; prostate TD1 0.586 / 0.769 / 0.790 / 0.817; prostate TD2 0.609 / 0.786 / 0.858 / 0.834; heart 0.670 / 0.744 / 0.742 / 0.806 (verified-from-source).

## Limitations
- Author-stated: slow, no convergence guarantee, DAE can lose contact with the image.
- Reviewer: mask-level shape prior needs labelled masks; gains concentrate where the shift is mostly intensity/contrast.

## Relevance to CD / CV / HEX
Our shift between sources is largely brightness and vignetting. The paper's core finding, that adapting only the first few layers that normalise intensity is safer than adapting the whole net (whole-net adaptation dropped Dice to 0.671), maps to us. The per-image optimisation (12 min) exceeds our 5-min T4 budget.

## What it suggests we try
Learn a light input-normalisation front-end (or apply flat-field correction) shared across sources rather than adapting backbone weights; validate on the old-vs-new source CD gap.

## Related notes
- [[adachi2025ssa]] — regression TTA updating only normalisation affine parameters
- [[zhang2020bigaug]] — augmentation-only route to unseen-domain robustness
- [[wang2021tent]] — normalisation-parameter TTA by entropy minimisation

## Sources
- https://ar5iv.arxiv.org/html/2004.04668
