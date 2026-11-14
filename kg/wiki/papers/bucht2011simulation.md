---
id: bucht2011simulation
type: paper
title: "Simulation of specular microscopy images of corneal endothelium, a tool for control of measurement errors"
authors: [Curry Bucht, Per Soderberg, Goran Manneberg]
year: 2011
venue: "Acta Ophthalmologica"
doi: "10.1111/j.1755-3768.2010.01974.x"
arxiv: ""
pmcid: ""
url: "https://pubmed.ncbi.nlm.nih.gov/20662798/"
citation: "Bucht C, Soderberg P, Manneberg G. Simulation of specular microscopy images of corneal endothelium, a tool for control of measurement errors. Acta Ophthalmol. 2011;89(3):e242-e250. doi:10.1111/j.1755-3768.2010.01974.x"
tier: core
group: t8-synthetic
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "A simulator generates specular-microscopy-like endothelium images with preset mean and variation of cell structure and stores exact per-cell morphometry; analysing simulated images with Topcon imagenet-640 revealed a scaling error in the software's cell-size-variation (CV) estimate and an operator-dependent random error."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Direct precedent that a Topcon device's CV is a scaled version of true CV; matches our fact that device CV is about twice the Voronoi area CV, and offers a synthetic-image route to calibrate that mapping."
code_release: unknown
weights_release: unknown
data_scale: "several hundred simulated images over the physiological ECD range; 12 analysed with imagenet-640 by one operator"
key_numbers:
  - "Simulator stores exact per-cell morphometry; image quality varied with a filter toolbox (from-abstract)"
  - "Several hundred unique images generated; 12 selected evenly over the physiological ECD range for the error study (from-abstract)"
  - "imagenet-640 estimates of cell size variation show a scaling error vs true simulated values; random error strongly operator-dependent (from-abstract)"
  - "Slope / intercept of the scaling error: not reported in what was read (from-abstract)"
results:
  - {task: "device CV vs true CV on simulated images", dataset: "12 simulated specular images", metric: "regression of software estimate on truth", value: "scaling error present (size not in abstract)", comparator: "true simulated morphometry", comparator_value: "reference", locator: "Abstract", provenance: from-abstract}
suggests_try: "Treat device CV as a scaled (and possibly offset) function of true area CV: fit a per-source linear/isotonic map from a segmentation-robust CV estimate (expert-marker or detector-based) to device CV on OOF and apply it before blending, rather than asking the network to learn the scale implicitly; widen predicted CV spread accordingly."
tags: [simulation, synthetic-images, measurement-error, topcon, imagenet, calibration]
search_keywords: [simulated specular microscopy images, corneal endothelium simulator, imagenet-640 scaling error]
uses_dataset: []
uses_method: [image simulation, regression calibration]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Only the abstract was accessible (Wiley 403; full text exists as a PDF on an Uppsala site, not fetched under the no-download rule); slope of the scaling error and simulator algorithm unverified", locator: "Abstract"}
  - {stated_by: reviewer, note: "12 images, one operator, semi-automatic imagenet-640 (old Topcon software); may not transfer to the CLEAR-EC device", locator: "Abstract"}
---

# Simulation of specular microscopy images of corneal endothelium

## TL;DR
Bucht et al. build software that synthesises specular microscopy images of the endothelium with a preset mean and variation of cell structure, store the exact morphometry of each cell, and degrade image quality with filters. Using 12 simulated images, they show that Topcon's imagenet-640 semi-automatic analysis estimates cell size variation with a scaling error, and that random error depends strongly on the operator.

## Citation and link
- Bucht C, Soderberg P, Manneberg G. Acta Ophthalmol. 2011;89(3):e242-e250. doi:10.1111/j.1755-3768.2010.01974.x
- https://pubmed.ncbi.nlm.nih.gov/20662798/ (abstract via PubMed metadata)

## Problem and motivation
Clinical morphometry has no ground truth; a simulator with known morphometry allows calibration across centres and quality control.

## Data
Several hundred simulated images spanning physiological ECD; 12 chosen evenly over ECD and analysed by one trained operator with imagenet-640.

## Method
- Randomised generation of endothelial mosaics with preset expected average and variation of cell structure (algorithm details not in abstract).
- Toolbox of image filters to approximate clinical image quality.
- Regression of software estimates (CV, HEX, ECD) on simulator truth.
- Same group: Fourier/diffraction-based fully automated morphometry (SPIE 2009, vol. 7163; SPIE 2010, vol. 7550, 292 images) with ECD correlating "fairly strong" with semi-automated analysis (search snippet only).

## Results
- Scaling error in the software's cell-size-variation estimate (from-abstract).
- Random error strongly operator-dependent (from-abstract).
- Numeric slopes: not reported in what was read.

## Limitations
- Author-stated: demonstration on 12 images only.
- Reviewer: abstract-level; old software; simulator realism unknown.

## Relevance to CD / CV / HEX
This is the closest published evidence for our finding that device CV is about twice the Voronoi area CV of the same cells: a Topcon algorithm's CV is systematically scaled relative to true cell-area CV. It implies that a model predicting "true" CV will be off by a scale factor unless calibrated, and that regression-to-the-mean (our CV predictions span only 66 % of label spread) may partly be a scale-compression problem.

## What it suggests we try
(1) Fit and apply an explicit scale map from robust CV features to device CV (per image source); (2) use synthetic mosaics with known CV, rendered to look like our frames, to check whether our CV head is scale-compressed (slope of predicted vs true CV < 1) and correct it with a post-hoc linear stretch on OOF.

## Related notes
- [[sierra2023sdm]] — Topcon CellCount CV +29.9 points vs manual, rising with CV
- [[viguerasguillen2019fcn]] — Topcon software CV relative error 10.1 % vs manual
- [[kucharski2023gan]] — modern synthetic training data for segmentation

## Sources
- https://pubmed.ncbi.nlm.nih.gov/20662798/ (via PubMed metadata tool)
- WebSearch snippets for the SPIE 2009/2010 Fourier morphometry papers (spiedigitallibrary.org)
