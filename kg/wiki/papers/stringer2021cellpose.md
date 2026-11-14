---
id: stringer2021cellpose
type: paper
title: "Cellpose: a generalist algorithm for cellular segmentation"
authors: [Carsen Stringer, Tim Wang, Michalis Michaelos, Marius Pachitariu]
year: 2021
venue: "Nature Methods"
doi: "10.1038/s41592-020-01018-x"
arxiv: ""
pmcid: ""
url: "https://pubmed.ncbi.nlm.nih.gov/33318659/"
citation: "Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods. 2021;18(1):100-106. doi:10.1038/s41592-020-01018-x"
tier: candidate
group: t4-cell-foundation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "Generalist cell segmenter: a U-Net predicts horizontal/vertical spatial gradients of a simulated diffusion from each cell centre plus cell probability; following the flows groups pixels into cells; trained on a varied dataset with over 70,000 objects, no retraining or parameter tuning needed."
relevance_clearec: medium
relevance_metrics: [CV, HEX]
relevance_note: "Defines the flow representation used by Cellpose-SAM (our zero-shot mask source and one regression backbone) and by MEDIAR; the 'diameter' input is the main knob when applying it to our ~25-30 px cells."
code_release: open
weights_release: open
data_scale: "Varied cell image dataset with over 70,000 segmented objects"
key_numbers:
  - "Training set contains over 70,000 segmented objects (from-abstract)"
  - "Segments a wide range of image types without model retraining or parameter adjustment (from-abstract)"
  - "3D extension reuses the 2D model, no 3D labels needed (from-abstract)"
results: []
suggests_try: "Nothing new beyond the Cellpose-SAM note; background for the flow representation."
tags: [cell-segmentation, generalist-model, gradient-flows, U-Net]
search_keywords: [Cellpose, cyto, gradient flow tracking, generalist cell segmentation]
uses_dataset: [Cellpose dataset]
uses_method: [U-Net, spatial gradient flows, flow tracking]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Abstract-level read only; no specular/endothelial images known in training set", locator: "abstract"}
---

# Cellpose: a generalist algorithm for cellular segmentation

## TL;DR
Cellpose introduced a generalist deep segmentation model that predicts spatial gradient flows toward cell centres and a cell probability map; pixels that converge to the same sink form one cell. Trained on a diverse set of cell images (over 70,000 objects), it works across many image types without retraining or tuning.

## Citation and link
- Stringer C, Wang T, Michaelos M, Pachitariu M. Nat Methods. 2021;18(1):100-106. doi:10.1038/s41592-020-01018-x
- https://pubmed.ncbi.nlm.nih.gov/33318659/ (abstract via PubMed metadata)

## Problem and motivation
Deep segmenters were specialised to single image types with large labelled sets.

## Data
New varied dataset of cell images with over 70,000 segmented objects (from abstract).

## Method
U-Net predicting x/y flows of a heat-diffusion field from each cell's centre plus a probability map; gradient tracking groups pixels; 3D by combining 2D predictions.

## Results
- Outperforms established methods on 2D and 3D datasets (from-abstract; numbers not read).

## Limitations
- Author-stated: not read (abstract only).
- Reviewer: generalist training set excludes specular endothelium.

## Relevance to CD / CV / HEX
Background for [[pachitariu2025cellposesam]], [[stringer2025cellpose3]], [[lee2022mediar]] and [[reith2025selfadapt]] (which adapts the flow outputs with uncertainty-masked MSE).

## What it suggests we try
Nothing actionable; background only.

## Related notes
- [[pachitariu2025cellposesam]] — SAM-encoder successor
- [[stringer2025cellpose3]] — restoration + cyto3 successor
- [[reith2025selfadapt]] — label-free adaptation of Cellpose models

## Sources
- https://pubmed.ncbi.nlm.nih.gov/33318659/
- PubMed metadata PMID 33318659
