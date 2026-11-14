---
id: nurzynska2024gan
type: paper
title: "Application Of Generative Adversarial Network for Data Augmentation and Multiplication to Automated Cell Segmentation of the Corneal Endothelium"
authors: [Karolina Nurzynska, Krystian Jandy, Pawel Weichbroth]
year: 2024
venue: "International Conference on Information Systems Development (ISD 2024), poster"
doi: "10.62036/ISD.2024.15"
arxiv: ""
pmcid: ""
url: "https://aisel.aisnet.org/isd2014/proceedings2024/datascience/4"
citation: "Nurzynska K, Jandy K, Weichbroth P. Application Of Generative Adversarial Network for Data Augmentation and Multiplication to Automated Cell Segmentation of the Corneal Endothelium. Proc ISD 2024, Gdansk. doi:10.62036/ISD.2024.15"
tier: candidate
group: t8-synthetic
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: workshop
tldr: "A GAN trained on the small Alizarine endothelium dataset generates over 513k synthetic image/mask pairs; a segmentation model trained on part of them reaches mean IoU 81 % across datasets (poster, visual assessment)."
relevance_clearec: low
relevance_metrics: [CV, HEX]
relevance_note: "Second example of GAN-multiplied endothelium training data; weak evidence (poster, mIoU only, stained ex vivo Alizarine images rather than specular)."
code_release: unknown
weights_release: unknown
data_scale: "Alizarine dataset (about 30 images typical of public sets) -> over 513,000 synthetic images"
key_numbers:
  - "Over 513k synthetic images generated from the Alizarine dataset (from-abstract)"
  - "Mean IoU 81 % averaged across datasets (from-abstract)"
  - "Public endothelium datasets average about 30 images (from-abstract)"
results:
  - {task: "endothelium segmentation trained on GAN data", dataset: "Alizarine + other datasets (unspecified)", metric: "mIoU", value: "81 %", comparator: "not reported", comparator_value: "not reported", locator: "Abstract", provenance: from-abstract}
suggests_try: "Nothing actionable on its own; it only confirms that GAN multiplication of a tiny labelled set is feasible, which supports trying a mosaic-conditioned generator for our point/edge model only if time remains after calibration work."
tags: [gan, synthetic-data, alizarine, data-multiplication]
search_keywords: [GAN data augmentation corneal endothelium, Alizarine dataset GAN]
uses_dataset: [Alizarine]
uses_method: [GAN, semantic segmentation]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Poster; GAN and segmenter architectures not given; no comparison with real-data training; Alizarine images are stained ex vivo, not specular", locator: "Abstract"}
---

# GAN for data augmentation and multiplication in corneal endothelium segmentation

## TL;DR
Nurzynska, Jandy and Weichbroth train a GAN on the Alizarine endothelium dataset and generate over 513k synthetic images with masks. A segmentation model trained on part of this set gets an average mIoU of 81 %.

## Citation and link
- Nurzynska K, Jandy K, Weichbroth P. ISD 2024 Proceedings (poster). doi:10.62036/ISD.2024.15
- https://aisel.aisnet.org/isd2014/proceedings2024/datascience/4

## Problem and motivation
Public endothelium datasets are tiny (about 30 images each).

## Data
Alizarine dataset (stained ex vivo endothelium) for GAN training; evaluation datasets not named in the abstract.

## Method
GAN generating image/mask pairs (architecture not stated); semantic segmentation model trained on a subset.

## Results
- > 513k synthetic images (from-abstract).
- mIoU 81 % across datasets (from-abstract).

## Limitations
- Author-stated: precision judged by visual inspection.
- Reviewer: no real-data baseline; no morphometry.

## Relevance to CD / CV / HEX
Weak; the synthetic route matters for us only as a source of geometry-labelled training images (see [[kucharski2023gan]]).

## What it suggests we try
Nothing actionable; background only.

## Related notes
- [[kucharski2023gan]] — stronger GAN study with mosaic-conditioned generation
- [[sanchez2024warping]] — geometric augmentation alternative

## Sources
- https://aisel.aisnet.org/isd2014/proceedings2024/datascience/4
