---
id: xie2018counting
type: paper
title: "Microscopy cell counting and detection with fully convolutional regression networks"
authors: [Weidi Xie, J. Alison Noble, Andrew Zisserman]
year: 2018
venue: "Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization"
doi: "10.1080/21681163.2016.1149104"
arxiv: ""
pmcid: ""
url: "https://api.crossref.org/works/10.1080/21681163.2016.1149104"
citation: "Xie W, Noble JA, Zisserman A. Microscopy cell counting and detection with fully convolutional regression networks. Comput Methods Biomech Biomed Eng Imaging Vis. 2018;6(3):283-292 (online 2016). doi:10.1080/21681163.2016.1149104"
tier: candidate
group: t5-weak-supervision
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "Fully convolutional regression networks predict a cell density map from dot annotations, counting clumped and overlapping cells where segmentation fails; networks trained only on synthetic images transfer to real microscopy, and fine-tuning improves them further."
relevance_clearec: medium
relevance_metrics: [CD]
relevance_note: "The standard deep density-map recipe for microscopy cells; the synthetic-to-real transfer claim supports training a density/centre head on synthetic endothelium plus the 385 Topcon point images."
code_release: unknown
weights_release: unknown
data_scale: "standard synthetic cell benchmarks plus real microscopy images (sizes not read)"
key_numbers:
  - "Density-map regression with two FCRN architectures accepting arbitrary input size (from-abstract)"
  - "Networks trained only on synthetic images perform well on real microscopy without fine-tuning; fine-tuning improves further (from-abstract)"
  - "State-of-the-art counting on standard synthetic benchmarks; exact errors not reported in abstract (from-abstract)"
results: []
suggests_try: "Treat a density head as the cheapest location-aware CD estimator: train on Gaussian-smoothed Topcon points (plus any synthetic endothelium from the t8 notes), integrate over the recorded analysis box, and use it as an ensemble member or feature for CD."
tags: [density-map, cell-counting, microscopy, FCRN, synthetic-to-real]
search_keywords: [FCRN cell counting, microscopy density map, cell detection regression network]
uses_dataset: [VGG synthetic cells]
uses_method: [density-map-regression, FCRN]
compares_with: []
extends: ["[[lempitsky2010counting]]"]
has_limitation:
  - {stated_by: reviewer, note: "Abstract-level only (publisher page 403); counts only, no morphometry", locator: "n/a"}
---

# Microscopy cell counting and detection with fully convolutional regression networks

## TL;DR
Xie, Noble and Zisserman bring deep FCNs to Lempitsky-style density counting in microscopy. The density map handles clumped and overlapping cells, its local maxima give detections, and models trained on synthetic data transfer to real images.

## Citation and link
- Xie W, Noble JA, Zisserman A. Comput Methods Biomech Biomed Eng Imaging Vis. 2018;6(3):283-292 (online 2 May 2016). doi:10.1080/21681163.2016.1149104.
- Read: Crossref metadata (https://api.crossref.org/works/10.1080/21681163.2016.1149104) and abstract via search listings; publisher page returned 403.

## Problem and motivation
Segmentation-based counting breaks on clumped or overlapping cells; a density regression avoids explicit instance separation.

## Data
Standard synthetic fluorescence cell benchmarks and real microscopy images (from-abstract); sizes not read.

## Method
Two fully convolutional regression networks (FCRN) that map an image of any size to a density map trained from dot annotations; detection from density maxima (from-abstract).

## Results
State of the art on synthetic benchmarks; synthetic-only training generalises to real data (from-abstract). Numbers not reported here.

## Limitations
- Author-stated: not read.
- Reviewer: abstract only; density maps do not give cell areas or neighbours.

## Relevance to CD / CV / HEX
CD only. Our point detector misses 15 % and adds 16.5 % false cells; a density integral averages these errors out instead of committing to discrete detections, which suits CD over a 538 x 408 px box.

## What it suggests we try
Same as [[lempitsky2010counting]]: density head on Topcon points, CD from box integral, compare OOF CD MAPE; low cost because the head can sit on an existing backbone.

## Related notes
- [[lempitsky2010counting]] — the density-counting formulation this paper deepens.
- [[wang2020dmcount]] — better loss for density heads.

## Sources
- https://api.crossref.org/works/10.1080/21681163.2016.1149104
- https://intelligent-earth.ox.ac.uk/node/3985971 (abstract via search snippet; page later 404)
