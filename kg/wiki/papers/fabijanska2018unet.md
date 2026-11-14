---
id: fabijanska2018unet
type: paper
title: "Segmentation of corneal endothelium images using a U-Net-based convolutional neural network"
authors: [Anna Fabijanska]
year: 2018
venue: "Artificial Intelligence in Medicine"
doi: "10.1016/j.artmed.2018.04.004"
arxiv: ""
pmcid: ""
url: "https://pubmed.ncbi.nlm.nih.gov/29680687/"
citation: "Fabijanska A. Segmentation of corneal endothelium images using a U-Net-based convolutional neural network. Artif Intell Med. 2018;88:1-13. doi:10.1016/j.artmed.2018.04.004"
tier: core
group: t1-endothelium-segmentation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "U-Net edge probability + binarisation + skeletonisation on 30 specular images: DICE 0.86, cell-count MAPE 4.5 %, but morphometric errors of 5.2 % (ECD), 6.2 % (HEX) and 11.93 % (CV), showing CV amplifies small segmentation errors the most."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Clean dose-response: a 4.5 % cell-count error already gives about 12 % CV error, close to our CV MAPE (10.11); supports not building CV on imperfect masks."
code_release: unknown
weights_release: unknown
data_scale: "30 corneal endothelium specular images (15 train / 15 test per later papers)"
key_numbers:
  - "AUROC 0.92 for border pixels; DICE 0.86 on average (from-abstract)"
  - "Cell-number MAPE 4.5 % (from-abstract)"
  - "Morphometric errors: ECD 5.2 %, hexagonality 6.2 %, CV of cell size 11.93 % (from-abstract)"
results:
  - {task: "morphometry from U-Net edges", dataset: "30 specular images", metric: "percentage error", value: "ECD 5.2 / HEX 6.2 / CV 11.93", comparator: "manual ground truth", comparator_value: "reference", locator: "Abstract", provenance: from-abstract}
suggests_try: "Use this as the expected CV penalty of segmentation-derived features: about 12 % CV error at only 4.5 % count error, versus our detector's 15 % miss / 16.5 % false rate; so route CV through image regression plus robust neighbour-distance statistics, and use mask area CV only as a weak, calibrated input."
tags: [u-net, edge-detection, skeletonisation, small-dataset]
search_keywords: [U-Net corneal endothelium segmentation, endothelial cell edge probability map, morphometric error CV hexagonality]
uses_dataset: []
uses_method: [U-Net, skeletonisation]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "30 images only; abstract-level reading; ground truth manual edges, not device values", locator: "Abstract"}
---

# Segmentation of corneal endothelium images using a U-Net-based CNN

## TL;DR
Fabijanska trains a U-Net to mark pixels on cell borders, then binarises and skeletonises the output into one-pixel edges. On 30 images the edges look good (AUROC 0.92, DICE 0.86, cell-count MAPE 4.5 %), yet the morphometric errors grow from 5.2 % for ECD to 6.2 % for HEX and 11.93 % for CV.

## Citation and link
- Fabijanska A. Artif Intell Med. 2018;88:1-13. doi:10.1016/j.artmed.2018.04.004
- https://pubmed.ncbi.nlm.nih.gov/29680687/ (abstract via PubMed metadata)

## Problem and motivation
Endothelial cells were extracted manually or semi-automatically; no automatic method was reliable.

## Data
30 corneal endothelial specular images with cells of different sizes; manual edges. Later papers report a 15/15 train/test split (search snippet).

## Method
U-Net trained to classify border pixels; edge probability map binarised and skeletonised to one-pixel edges; ECD, HEX and CV computed from the resulting cells.

## Results
- AUROC 0.92; DICE 0.86 (from-abstract).
- Cell-count MAPE 4.5 % (from-abstract).
- ECD 5.2 %, HEX 6.2 %, CV 11.93 % error (from-abstract).

## Limitations
- Author-stated: edges require a limited number of manual corrections.
- Reviewer: very small set; error type (MAPE vs relative) for morphometrics as stated in abstract only.

## Relevance to CD / CV / HEX
Shows the amplification order ECD < HEX < CV when masks are imperfect. Matches our finding that zero-shot masks give area CV r 0.38 and that misses ruin area CV, and explains why our best CV pipeline still regresses to the mean.

## What it suggests we try
Do not invest remaining days in better mask-to-CV pipelines unless the miss rate can drop below a few percent; instead add segmentation-robust CV proxies (neighbour-distance CV, local spacing dispersion) and target the CV tails with distribution-aware losses.

## Related notes
- [[viguerasguillen2019fcn]] — under 1 % wrong cells gives ~3 % CV error
- [[kucharski2021cnnwatershed]] — follow-up by the same lab
- [[kucharski2023gan]] — synthetic training data by the same lab

## Sources
- https://pubmed.ncbi.nlm.nih.gov/29680687/ (via PubMed metadata tool)
