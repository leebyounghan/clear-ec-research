---
id: viguerasguillen2019regression
type: paper
title: "Convolutional neural network-based regression for biomarker estimation in corneal endothelium microscopy images"
authors: [Juan P. Vigueras-Guillen, Jeroen van Rooij, Hans G. Lemij, Koenraad A. Vermeer, Lucas J. van Vliet]
year: 2019
venue: "41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC 2019)"
doi: "10.1109/EMBC.2019.8857201"
arxiv: ""
pmcid: ""
url: "https://research.tudelft.nl/en/publications/convolutional-neural-network-based-regression-for-biomarker-estim/"
citation: "Vigueras-Guillen JP, van Rooij J, Lemij HG, Vermeer KA, van Vliet LJ. Convolutional neural network-based regression for biomarker estimation in corneal endothelium microscopy images. Proc IEEE EMBC 2019:876-881. doi:10.1109/EMBC.2019.8857201"
tier: core
group: t1-endothelium-segmentation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "A CNN regresses ECD, CV and HEX directly from CNN edge-probability images of 738 specular images; MAPE 4.32 % ECD, 7.07 % CV, 11.74 % HEX, worse than the authors' segmentation pipeline, with HEX the hardest target for direct regression."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Closest prior to CLEAR-EC's black-box regression; it regresses from an edge map (not raw pixels) and finds HEX the hardest to regress directly."
code_release: unknown
weights_release: unknown
data_scale: "738 Topcon SP-1P images (Baerveldt glaucoma study + DSAEK clinical care, Rotterdam Eye Hospital)"
key_numbers:
  - "Direct CNN regression MAPE: ECD 4.32 %, CV 7.07 %, HEX 11.74 % (from-abstract)"
  - "738 images, all unhealthy endothelium (Baerveldt study + DSAEK) (from-abstract)"
  - "Using the regressor to tune per-image postprocessing gave no clear benefit (from-abstract)"
results:
  - {task: "direct regression of ECD/CV/HEX from edge images", dataset: "738 Rotterdam specular images", metric: "MAPE", value: "4.32 / 7.07 / 11.74", comparator: "authors' segmentation pipeline", comparator_value: "not reported in abstract (stated as better)", locator: "Abstract", provenance: from-abstract}
suggests_try: "Feed the regression head a predicted edge/boundary-probability map (or our cell-center heatmap) as an extra input channel instead of raw pixels only; the paper's edge-input regressor got CV MAPE 7.07 % vs our raw-image OOF CV 10.11 (different labels, so treat as a hint, not a target)."
tags: [direct-regression, edge-map, topcon, morphometry]
search_keywords: [CNN regression endothelial cell density, biomarker regression corneal endothelium, segmentation-free CV HEX]
uses_dataset: [Rotterdam Eye Hospital Baerveldt and DSAEK images]
uses_method: [CNN regression, CNN edge detection]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Does not outperform the segmentation-based pipeline", locator: "Abstract"}
  - {stated_by: reviewer, note: "Targets are manual-segmentation morphometrics on small crops, not device labels; only the abstract was readable here", locator: "Abstract"}
---

# CNN-based regression for biomarker estimation in corneal endothelium images

## TL;DR
Instead of postprocessing CNN edge maps into a segmentation, a second CNN regresses ECD, CV and HEX directly from the edge image. It reaches MAPE 4.32 / 7.07 / 11.74 % (ECD / CV / HEX) but does not beat the segmentation pipeline. HEX is the metric that suffers most from skipping explicit cell topology.

## Citation and link
- Vigueras-Guillen JP, et al. Proc IEEE EMBC 2019:876-881. doi:10.1109/EMBC.2019.8857201
- https://research.tudelft.nl/en/publications/convolutional-neural-network-based-regression-for-biomarker-estim/

## Problem and motivation
The earlier pipeline (edge CNN, ROI detection, postprocessing, false-edge refinement) is long; a regressor could simplify it.

## Data
Topcon SP-1P specular images, 738 images from the Baerveldt glaucoma-device study and DSAEK clinical care at the Rotterdam Eye Hospital; all unhealthy endothelium. Split and annotation details not reported in the abstract.

## Method
- Input: edge-probability images produced by the authors' CNN-Edge, not raw intensity.
- Output: ECD, CV, HEX by regression.
- Second experiment: the regressor's estimates used to tune per-image postprocessing parameters.

## Results
- MAPE ECD 4.32 %, CV 7.07 %, HEX 11.74 % (from-abstract).
- Regression did not outperform the authors' previous segmentation pipeline (from-abstract).
- Tuning postprocessing with the regressor brought no clear benefit (from-abstract).

## Limitations
- Author-stated: worse than segmentation-based estimation.
- Reviewer: labels are manual-segmentation values on short crops, while CLEAR-EC labels are device values on a technician-chosen box; abstract only.

## Relevance to CD / CV / HEX
The ordering (ECD easiest, HEX hardest for regression) matches the intuition that HEX needs topology. In CLEAR-EC, HEX label tracks the six-neighbour share of expert markers within 1-2 %, so a neighbour-graph feature from a point detector may beat image regression for HEX, while CV regression may profit from an edge-map input.

## What it suggests we try
Add a boundary/edge-probability channel (from any zero-shot edge or Cellpose flow output) to the regression backbone input and compare CV OOF against the raw-image model; for HEX, add the six-neighbour share from Delaunay of detected centers in the label box as a feature.

## Related notes
- [[viguerasguillen2020dsaek]] — the segmentation pipeline this was compared against
- [[viguerasguillen2019fcn]] — the edge CNN that produces the input

## Sources
- https://research.tudelft.nl/en/publications/convolutional-neural-network-based-regression-for-biomarker-estim/
