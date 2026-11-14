---
id: viguerasguillen2018merging
type: paper
title: "Corneal Endothelial Cell Segmentation by Classifier-Driven Merging of Oversegmented Images"
authors: [Juan P. Vigueras-Guillen, Esma Islamaj, Jeroen van Rooij, Hans G. Lemij, Lucas J. van Vliet, Koenraad A. Vermeer, et al.]
year: 2018
venue: "IEEE Transactions on Medical Imaging"
doi: "10.1109/TMI.2018.2841910"
arxiv: ""
pmcid: ""
url: "https://repository.tudelft.nl/record/uuid:ed324a6b-d219-4f74-bc9e-bc0475ce84a9"
citation: "Vigueras-Guillen JP, et al. Corneal Endothelial Cell Segmentation by Classifier-Driven Merging of Oversegmented Images. IEEE Trans Med Imaging. 2018;37(10):2278-2289. doi:10.1109/TMI.2018.2841910"
tier: candidate
group: t1-endothelium-segmentation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "Stochastic-watershed superpixels are merged by SVMs using intensity and shape features; on Topcon SP-1P images 95.8 % of cells are correctly merged and 2.0 % undersegmented, with more precise ECD/CV/HEX than the vendor software."
relevance_clearec: low
relevance_metrics: [CD, CV, HEX]
relevance_note: "Classical oversegment-then-merge pipeline; relevant only as a way to fix merges/splits from an imperfect detector before computing area statistics."
code_release: unknown
weights_release: unknown
data_scale: "Topcon SP-1P in vivo set plus confocal, phase-contrast and fluorescence datasets (sizes not read)"
key_numbers:
  - "95.8 % correctly merged cells, 2.0 % undersegmented cells on Topcon SP-1P (from-abstract)"
  - "Parameter estimates more precise than vendor software with similar or better accuracy (from-abstract)"
results:
  - {task: "cell segmentation", dataset: "Topcon SP-1P in vivo images", metric: "correctly merged cells", value: "95.8 %", comparator: "not reported in abstract", comparator_value: "not reported", locator: "Abstract", provenance: from-abstract}
suggests_try: "Nothing actionable in the time left; if masks are used, an oversegment-then-merge step with a cell-size prior from the Fourier peak could cut merged cells, but the 2.0 % undersegmentation here is still far from our detector's miss rate problem."
tags: [superpixels, stochastic-watershed, svm, merging, topcon, classical]
search_keywords: [classifier-driven merging superpixels endothelium, stochastic watershed oversegmentation]
uses_dataset: [Rotterdam Topcon SP-1P images]
uses_method: [stochastic watershed, SVM, superpixel merging]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Abstract-level only; full text is a PDF in the TU Delft repository, not fetched under the no-download rule", locator: "n/a"}
---

# Corneal endothelial cell segmentation by classifier-driven merging of oversegmented images

## TL;DR
The Rotterdam/Delft group oversegments specular images into superpixels with a stochastic watershed and lets SVM classifiers decide which superpixels belong to the same cell. On Topcon SP-1P images 95.8 % of cells are merged correctly and 2.0 % are undersegmented; clinical parameters are more precise than the vendor software.

## Citation and link
- Vigueras-Guillen JP, et al. IEEE Trans Med Imaging. 2018;37(10):2278-2289. doi:10.1109/TMI.2018.2841910
- https://repository.tudelft.nl/record/uuid:ed324a6b-d219-4f74-bc9e-bc0475ce84a9 (record; abstract via search snippets)

## Problem and motivation
Low contrast and large cell-size variation defeat single-pass watershed methods.

## Data
Topcon SP-1P in vivo specular images; also tested on confocal, phase-contrast and fluorescence endothelium datasets.

## Method
Stochastic watershed oversegmentation; intensity and shape features per superpixel pair; SVMs decide merges.

## Results
- 95.8 % correctly merged, 2.0 % undersegmented cells (from-abstract).
- More precise parameters than vendor software (from-abstract).

## Limitations
- Author-stated: not available (abstract only).
- Reviewer: superseded by the group's CNN pipelines ([[viguerasguillen2019fcn]]).

## Relevance to CD / CV / HEX
Background on why merges dominate CV error and how a merge classifier can control them.

## What it suggests we try
Nothing actionable; background only.

## Related notes
- [[viguerasguillen2019fcn]] — CNN successor
- [[piorkowski2017segmentation]] — segmentation choice vs parameters

## Sources
- https://repository.tudelft.nl/record/uuid:ed324a6b-d219-4f74-bc9e-bc0475ce84a9
- WebSearch snippets of the abstract
