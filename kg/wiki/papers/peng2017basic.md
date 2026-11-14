---
id: peng2017basic
type: paper
title: "A BaSiC tool for background and shading correction of optical microscopy images"
authors: [Tingying Peng, Kurt Thorn, Timm Schroeder, Lichao Wang, Fabian J. Theis, Carsten Marr, et al.]
year: 2017
venue: "Nature Communications"
doi: "10.1038/ncomms14836"
arxiv: ""
pmcid: "PMC5472168"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC5472168/"
citation: "Peng T, Thorn K, Schroeder T, Wang L, Theis FJ, Marr C, Navab N. A BaSiC tool for background and shading correction of optical microscopy images. Nat Commun. 2017;8:14836. doi:10.1038/ncomms14836"
tier: core
group: t7-domain-shift
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "Retrospective flat-field (multiplicative shading) and dark-field estimation from a stack of images of the same optical system via rank-2 low-rank + sparse decomposition with Fourier smoothness; works from as few as five images and matches prospective (reference-slide) correction at about 100 images."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Our 2023-24 source is darker and vignetted; thousands of frames per source with uncorrelated cell mosaics are exactly BaSiC's assumption, so a per-source flat-field can be estimated without any reference image or annotation and removed before segmentation or regression."
code_release: open
weights_release: none
data_scale: "Synthetic stacks, 45 real whole-slide images, time-lapse haematopoietic stem cell data"
key_numbers:
  - "Improves image quality with as few as five images; about equal to prospective correction at about 100 images (verified-from-source)"
  - "Significantly better than other retrospective methods with fewer than 500 images (P < 0.023, paired Wilcoxon, Bonferroni) (verified-from-source)"
  - "On synthetic data CIDRE needed about 100 images to reach the score BaSiC reached with about 10 (verified-from-source)"
  - "Cell-culture WSI: std of mean cell intensity across regions fell to about 20 % of uncorrected after correction (verified-from-source)"
  - "Segmentation/counting accuracy after correction: not reported (verified-from-source)"
results:
  - {task: "shading estimation", dataset: "45 real WSIs", metric: "median estimation score", value: "lowest of compared methods, no outliers", comparator: "CIDRE and others", comparator_value: "not reported numerically", locator: "Fig 2e", provenance: "verified-from-source"}
suggests_try: "Estimate one flat-field (and dark-field) per image source/ID format with BaSiC(Py) on a few hundred frames, divide it out before the point detector, Cellpose-SAM and the regressors, and compare per-source CD/CV/HEX MAPE (old 7.11 vs new 5.50 CD) with and without correction."
tags: [flat-field-correction, vignetting, shading, illumination, preprocessing, domain-shift]
search_keywords: [BaSiC, BaSiCPy, flat-field correction, retrospective shading correction, vignetting correction microscopy]
uses_dataset: []
uses_method: [low-rank plus sparse decomposition, reweighted L1, Fourier smoothness]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Assumes foreground is uncorrelated across images; static objects can be absorbed into the shading estimate; inspect the estimated profile", locator: "Discussion"}
  - {stated_by: author, note: "Per-image baseline may converge to foreground when foreground covers more than 50 % of the image", locator: "Discussion"}
  - {stated_by: reviewer, note: "Validated on fluorescence/brightfield; specular microscopy has fully tiled frames and a technician-chosen focus band, so the flat-field may mix optical vignetting with corneal curvature effects", locator: "n/a"}
---

# A BaSiC tool for background and shading correction of optical microscopy images

## TL;DR
BaSiC models each image as true signal times a flat-field plus a dark-field, and estimates both from a set of images taken with the same optics by decomposing the sorted pixel-intensity matrix into a rank-2 component and a sparse residual, with smoothness imposed in the Fourier domain and automatic regularisation. It needs few images (works from five), is robust to artefacts and outperforms other retrospective methods.

## Citation and link
- Peng T, Thorn K, Schroeder T, Wang L, Theis FJ, Marr C, Navab N. Nat Commun. 2017;8:14836. doi:10.1038/ncomms14836
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5472168/ ; code: Fiji plugin, MATLAB (marrlab/basic), BaSiCPy

## Problem and motivation
Uneven illumination (vignetting) biases intensity-based quantification and stitching; prospective correction needs reference images that are often not recorded.

## Data
Synthetic shading stacks; 45 real whole-slide images; time-lapse stem-cell imaging.

## Method
Rank-2 low-rank + sparse decomposition of the image stack, reweighted L1, linearised augmented Lagrangian; optional per-image baseline for temporal drift.

## Results
- Works from 5 images; matches prospective correction at about 100 images; beats other retrospective tools below 500 images (P < 0.023) (verified-from-source).
- Intensity std across regions down to about 20 % of uncorrected (verified-from-source).
- Biological effect (PU.1 change) visible only after correction (P = 1.2e-3) (verified-from-source).
- No segmentation/counting accuracy reported.

## Limitations
- Author-stated: correlated foreground, baseline drift to foreground, no photobleaching correction.
- Reviewer: specular images are reflected-light with dark cell borders; the estimated profile should be checked visually per source.

## Relevance to CD / CV / HEX
The old/new image sources differ in brightness and vignetting (CD error 7.11 vs 5.50). Removing a per-source shading field is a deterministic, annotation-free, test-time-cheap normalisation that can help both mask-based features (border contrast at frame edges controls merges, which ruin area CV) and black-box regressors. It is the concrete answer to "does restoration help darker vignetted images": correct the shading explicitly rather than with Cellpose3 restoration ([[stringer2025cellpose3]]).

## What it suggests we try
Per-source BaSiC flat-field, divided out at train and test time; check per-source MAPE and the zero-shot mask area-CV correlation (currently r 0.38) before and after.

## Related notes
- [[stringer2025cellpose3]] — restoration that does not target shading
- [[zhang2020bigaug]] — the augmentation alternative (train-time robustness instead of correction)
- [[karani2021tta]] — learned input normalisation at test time

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC5472168/
