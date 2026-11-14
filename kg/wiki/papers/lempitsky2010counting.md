---
id: lempitsky2010counting
type: paper
title: "Learning to count objects in images"
authors: [Victor Lempitsky, Andrew Zisserman]
year: 2010
venue: "Advances in Neural Information Processing Systems 23 (NIPS 2010)"
doi: ""
arxiv: ""
pmcid: ""
url: "https://ora.ox.ac.uk/objects/uuid:c75861b1-cd39-4929-be13-3b3446a1efe9"
citation: "Lempitsky V, Zisserman A. Learning to count objects in images. Advances in Neural Information Processing Systems 23 (NIPS 2010). 2010:1324-1332"
tier: candidate
group: t5-weak-supervision
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "Founding paper of density-map counting: learn a per-pixel density from dot annotations so that its integral over any region equals the object count, avoiding instance detection; trained with the MESA loss via a convex QP."
relevance_clearec: medium
relevance_metrics: [CD]
relevance_note: "CD is a count per area; a density map integrates to a count in any region, including the technician's analysis box, and is robust to the merge/split errors that hurt our point detector."
code_release: unknown
weights_release: none
data_scale: "VGG synthetic cell dataset, 200 images (100 train/validation, 100 test); UCSD pedestrians"
key_numbers:
  - "Density integral over any image region gives the object count in that region (from-abstract)"
  - "Loss (MESA) computed with a maximum-subarray algorithm; learning = convex QP with cutting planes (from-abstract)"
  - "Released cell dataset: 200 images, first 100 for training/validation, second 100 for testing (from-abstract)"
results: []
suggests_try: "Predict a cell-density map (trained on Gaussian-smoothed Topcon points) and read CD as density integral over the recorded analysis box; compare its CD MAPE against counting detected points in the same box, because density maps tolerate the 15 % misses / 16.5 % false detections better than discrete counts."
tags: [density-map, counting, dot-annotation, foundational]
search_keywords: [learning to count objects, density map counting, MESA distance]
uses_dataset: [VGG cells, UCSD pedestrians]
uses_method: [density-map-regression, MESA-loss]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Abstract-level only (proceedings pages and PDF unreachable); numbers not checked", locator: "n/a"}
  - {stated_by: reviewer, note: "Gives counts, not per-cell areas or neighbours, so says nothing about CV or HEX", locator: "Method"}
---

# Learning to count objects in images

## TL;DR
Counting by density estimation: from one dot per object, learn an image density whose integral over any region is the count there. No detection or segmentation is needed, which makes counting robust in crowded scenes.

## Citation and link
- Lempitsky V, Zisserman A. NIPS 2010, Advances in Neural Information Processing Systems 23, pp. 1324-1332.
- Read: https://ora.ox.ac.uk/objects/uuid:c75861b1-cd39-4929-be13-3b3446a1efe9 (abstract record).

## Problem and motivation
Counting by detection fails when objects overlap or touch; counting by global regression ignores the spatial annotation. The density formulation uses dots and supports counts in arbitrary sub-regions.

## Data
VGG synthetic fluorescence cell dataset (200 images, 100/100 split) and UCSD pedestrian video (from-abstract / search snippet).

## Method
Linear density from dense pixel features; MESA loss (max over subarrays of the absolute difference between predicted and ground-truth density integrals); regularised convex QP via cutting planes (from-abstract).

## Results
Counting errors not read (proceedings pages and PDF failed to load): not reported here.

## Limitations
- Author-stated: not read.
- Reviewer: abstract only; pre-deep-learning features.

## Relevance to CD / CV / HEX
CD in cells/mm2 is a density. The label is computed in a technician-chosen box of about 17 % of the frame; a density map lets us integrate over that recorded box (538 x 408 px) exactly, matching the label's sampling region, and tolerates touching cells that cause merges in point detection.

## What it suggests we try
Add a density head trained on Gaussian-smoothed Topcon points; CD_pred = integral over recorded box / box area; evaluate CD MAPE vs the current CD head and vs point counting, and also as an extra feature for the CD regressor.

## Related notes
- [[xie2018counting]] — deep FCN density regression for microscopy cells.
- [[wang2020dmcount]] — optimal-transport counting loss without Gaussian smoothing.
- [[kervadec2019constrained]] — image-level count/size as a constraint on such maps.

## Sources
- https://ora.ox.ac.uk/objects/uuid:c75861b1-cd39-4929-be13-3b3446a1efe9
