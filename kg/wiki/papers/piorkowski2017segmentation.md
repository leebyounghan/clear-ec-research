---
id: piorkowski2017segmentation
type: paper
title: "Influence of applied corneal endothelium image segmentation techniques on the clinical parameters"
authors: [Adam Piorkowski, Karolina Nurzynska, Jolanta Gronkowska-Serafin, Bettina Selig, Cezary Boldak, Daniel Reska]
year: 2017
venue: "Computerized Medical Imaging and Graphics"
doi: "10.1016/j.compmedimag.2016.07.010"
arxiv: ""
pmcid: ""
url: "https://badap.agh.edu.pl/publikacja/104675"
citation: "Piorkowski A, Nurzynska K, Gronkowska-Serafin J, Selig B, Boldak C, Reska D. Influence of applied corneal endothelium image segmentation techniques on the clinical parameters. Comput Med Imaging Graph. 2017;55:13-27. doi:10.1016/j.compmedimag.2016.07.010"
tier: core
group: t1-endothelium-segmentation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "Four segmentation methods (stochastic watershed, KH, active contours, Topcon ImageNET) give closely matching CV and side-length CV but noticeably different hexagonality on specular images; a border-unification framework reduces the spread."
relevance_clearec: high
relevance_metrics: [CV, HEX]
relevance_note: "Direct evidence on how much the segmentation method itself moves CV vs HEX; HEX is the method-sensitive metric, CV less so when segmentations are complete."
code_release: unknown
weights_release: unknown
data_scale: "not reported in abstract"
key_numbers:
  - "Four methods compared: FRSW, KH, SNAKE, Topcon ImageNET (from-abstract)"
  - "CV and CVSL in high accordance across methods; hexagonality shows higher variation (from-abstract)"
  - "Choice of method did not change the ability to discriminate images by clinical parameters (from-abstract)"
  - "Per-method numeric differences: not reported in abstract (from-abstract)"
results:
  - {task: "CV/CVSL/HEX agreement across segmentation methods", dataset: "specular microscopy images (size not reported in abstract)", metric: "qualitative accordance", value: "CV, CVSL high accordance; HEX higher variation", comparator: "with vs without border-unification framework", comparator_value: "framework lowers dispersion", locator: "Abstract", provenance: from-abstract}
suggests_try: "Add CV of cell side length / neighbour distance (CVSL-like) computed from detected centers as a CV feature, since side-length CV is as stable across segmentations as area CV; treat HEX from any single segmentation as method-dependent and calibrate it on the label scale rather than using it raw."
tags: [segmentation-comparison, watershed, active-contours, topcon, hexagonality]
search_keywords: [segmentation technique influence clinical parameters endothelium, CVSL coefficient of variation side length, Topcon ImageNET comparison]
uses_dataset: []
uses_method: [fast robust stochastic watershed, KH method, active contours, Topcon ImageNET]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Only the abstract was accessible (paywalled); sample size and numeric spreads not verified", locator: "Abstract"}
  - {stated_by: reviewer, note: "All four methods are endothelium-tuned and produce near-complete segmentations; it does not address heavy miss/merge rates like ours", locator: "Abstract"}
---

# Influence of applied corneal endothelium image segmentation techniques on the clinical parameters

## TL;DR
Piorkowski and colleagues segment the same specular images with four methods (FRSW, KH, SNAKE, Topcon ImageNET) and compare the resulting clinical parameters. CV and side-length CV agree closely across methods, while hexagonality varies more. A framework that unifies the precise border location lowers the dispersion.

## Citation and link
- Piorkowski A, et al. Comput Med Imaging Graph. 2017;55:13-27. doi:10.1016/j.compmedimag.2016.07.010
- https://badap.agh.edu.pl/publikacja/104675 (bibliographic record with abstract)

## Problem and motivation
Specular images of living, unstained endothelium are low quality; clinical parameters depend on the chosen segmentation, and it was unclear by how much.

## Data
In vivo specular microscopy images; number of images and device details not reported in the abstract.

## Method
- Segmentations from fast robust stochastic watershed (FRSW), the KH method, an active-contour solution (SNAKE) and Topcon ImageNET.
- An "improvement framework" that relocates borders consistently regardless of preprocessing.
- Clinical parameters: CD, CV (cell area), CVSL (cell side length), H (hexagonality).

## Results
- CV and CVSL in high accordance across methods (from-abstract).
- H shows higher variation across methods (from-abstract).
- The framework improves repeatability of border location and reduces parameter dispersion (from-abstract).
- Method choice did not affect discriminating images by medical parameters (from-abstract).
- Numeric agreement values: not reported in what was read.

## Limitations
- Author-stated: not available (abstract only).
- Reviewer: no numbers verified; the methods compared all produce near-complete tessellations, so the result says nothing about segmentations with 15 % missed cells.

## Relevance to CD / CV / HEX
Supports ranking HEX as the most segmentation-sensitive metric, and side-length CV as an alternative polymegethism measure. In CLEAR-EC the HEX label tracks the expert six-neighbour share closely, so the HEX sensitivity here argues for a neighbour graph built from good centers rather than from masks. The CV stability here (complete, tuned segmentations) contrasts with our zero-shot masks (area CV r 0.38), suggesting the problem is incomplete/merged segmentation, not the CV definition.

## What it suggests we try
Compute side-length/neighbour-distance CV from Voronoi edges of detected centers inside the label box and use it as a feature for the CV head alongside area CV; it should degrade less under misses, as our own neighbour-distance CV result suggests.

## Related notes
- [[viguerasguillen2019fcn]] — dose-response of wrong cells on CV/HEX error
- [[scarpa2016morphometry]] — classical automated morphometry vs manual

## Sources
- https://badap.agh.edu.pl/publikacja/104675
- WebSearch snippets of the abstract (read.qxmd.com record 27553657)
