---
id: viguerasguillen2022feedbackattn
type: paper
title: "DenseUNets with feedback non-local attention for the segmentation of specular microscopy images of the corneal endothelium with guttae"
authors: [Juan P. Vigueras-Guillen, Jeroen van Rooij, Bart T. H. van Dooren, Hans G. Lemij, Esma Islamaj, Lucas J. van Vliet, et al.]
year: 2022
venue: "Scientific Reports"
doi: "10.1038/s41598-022-18180-1"
arxiv: "2203.01882"
pmcid: "PMC9388684"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC9388684"
citation: "Vigueras-Guillen JP, van Rooij J, van Dooren BTH, Lemij HG, Islamaj E, van Vliet LJ, Vermeer KA. DenseUNets with feedback non-local attention for the segmentation of specular microscopy images of the corneal endothelium with guttae. Sci Rep. 2022;12:14035. doi:10.1038/s41598-022-18180-1"
tier: core
group: t1-endothelium-segmentation
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc, abstract]
evidence: peer-reviewed
tldr: "On 1203 Topcon SP-1P images (500 with guttae), a 0.43 M-parameter DenseUNet with feedback non-local attention plus a cell-body network and Fourier-scaled watershed reaches MAE 23.16 cells/mm2 (ECD), 1.28 (CV) and 3.13 (HEX) vs manual, 3-6x lower than the Topcon software."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Largest Topcon specular set with manual masks; quantifies vendor-software error vs manual on CV/HEX, HEX definition sensitivity (vertex vs neighbour) and HEX underestimation below 100 cells."
code_release: open
weights_release: unknown
data_scale: "1203 Topcon SP-1P images (Baerveldt study 820 + UT-DSAEK 383), 500 with guttae, manual segmentation"
key_numbers:
  - "Best network (DenseUNet fNLA Mul) MAE vs manual: ECD 23.16 cells/mm2, CV 1.28, HEX 3.13 (verified-from-source)"
  - "Topcon software MAE vs manual, guttae / no guttae: ECD 166.81 / 118.74, CV 4.54 / 4.19, HEX 9.31 / 8.34 (verified-from-source)"
  - "CNN-Body framework, no guttae: ECD 17.64, CV 0.98, HEX 2.50 MAE; guttae: 34.66, 1.98, 4.58 (verified-from-source)"
  - "HEX by vertex method MAE 3.14 vs neighbour method 4.13 (verified-from-source)"
  - "Topcon detected 53 (guttae) / 77 (clean) cells per image vs manual 148 / 166 (verified-from-source)"
  - "HEX underestimated in images with fewer than 100 cells; guttae matter mainly below 25 cells (verified-from-source)"
results:
  - {task: "ECD/CV/HEX from segmentation", dataset: "1203 Topcon SP-1P images, 10-fold CV", metric: "MAE", value: "23.16 / 1.28 / 3.13", comparator: "plain UNet", comparator_value: "25.79 / 1.52 / 3.89", locator: "Table 1", provenance: verified-from-source}
  - {task: "ECD/CV/HEX, no-guttae images", dataset: "703 clean images", metric: "MAE", value: "17.64 / 0.98 / 2.50 (CNN-Body)", comparator: "Topcon software", comparator_value: "118.74 / 4.19 / 8.34", locator: "Table 2", provenance: verified-from-source}
suggests_try: "Copy two tricks that need no new labels: (1) stratified batches that oversample hard frames (here guttae; for us dark 2023-24 vignetted frames and outer CV deciles), which the authors call key; (2) a cell-body (inside-cell) probability head with a 0.5-0.75 threshold to select trustworthy cells before computing area statistics."
superseded: "2026-11-09: segmentation / self-training route closed for CV and HEX by the root-cause analysis; see body"
tags: [denseunet, attention, guttae, fuchs, topcon, cell-body, watershed]
search_keywords: [feedback non-local attention, guttae specular microscopy deep learning, DenseUNet endothelium]
uses_dataset: [Rotterdam Baerveldt study, Rotterdam UT-DSAEK study]
uses_method: [DenseUNet, non-local attention, Fourier cell-size estimation, watershed]
compares_with:
  - {target: "[[viguerasguillen2020dsaek]]", result: better, metric: "ECD/CV/HEX MAE", note: "CNN-Body beats the earlier CNN-ROI selection (P < 0.0001)", locator: "Table 2"}
extends: ["[[viguerasguillen2020dsaek]]"]
has_limitation:
  - {stated_by: author, note: "HEX still differs significantly from manual (Kruskal-Wallis P < 0.001); peripheral vertex detection sometimes faulty", locator: "Results/Discussion"}
  - {stated_by: author, note: "Data only on request", locator: "Data availability"}
  - {stated_by: reviewer, note: "Text says 500/703 guttae split but Fig. 5 shows 483/702", locator: "Fig. 5"}
---

# DenseUNets with feedback non-local attention for specular images with guttae

## TL;DR
The Rotterdam group scales its edge-plus-postprocessing pipeline to 1203 Topcon SP-1P images and adds feedback non-local attention and a cell-body network to handle guttae. Biomarker MAE vs manual drops to 23 cells/mm2, 1.3 CV points and 3.1 HEX points, 3-6x lower than Topcon's software. HEX remains the hardest metric and depends on how six-sidedness is counted.

## Citation and link
- Vigueras-Guillen JP, et al. Sci Rep. 2022;12:14035. doi:10.1038/s41598-022-18180-1 (preprint arXiv:2203.01882)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9388684 ; code: github.com/jpviguerasguillen/feedback-non-local-attention-fNLA

## Problem and motivation
Guttae (dark spots in Fuchs dystrophy) occlude cell edges and break vendor software; clinicians need ECD/CV/HEX from these images.

## Data
Topcon SP-1P, 240 x 528 px (0.25 x 0.55 mm). Baerveldt study 820 images (420 with guttae + 400 without) and UT-DSAEK 383 images (80 with guttae): 1203 total, 500 with guttae, graded low/medium/high. Manual segmentation (GIMP) with partial cells and discard areas; Topcon IMAGEnet values collected. Ten-fold CV by eye.

## Method
- CNN-Edge with Gaussian-blurred edge targets; CNN-Body (cell interiors) replaces the earlier CNN-ROI for selecting trustworthy cells.
- DenseUNet (0.38 M params) + fNLA attention fed from lower-resolution stages (0.43 M params total).
- Postprocessing: Fourier mean cell size, Gaussian smoothing sigma = 0.2 x cell size, watershed, removal of short/weak edges and border superpixels, keep superpixels with body probability > 0.5 (0.75 for fNLA).
- CV = std/mean of cell area; HEX = share of cells with six vertices (vertex method), all cells.
- Batches of 15 with 6 guttae images stratified by complexity; flips + elastic deformation; cross-entropy; Nadam.

## Results
- Table 1 MAE (ECD / CV / HEX): UNet 25.79 / 1.52 / 3.89; DenseUNet 23.81 / 1.30 / 3.36; DenseUNet fNLA Mul 23.16 / 1.28 / 3.13 (verified-from-source).
- Table 2 Topcon vs CNN-Body: guttae 166.81 / 4.54 / 9.31 vs 34.66 / 1.98 / 4.58; clean 118.74 / 4.19 / 8.34 vs 17.64 / 0.98 / 2.50 (verified-from-source).
- Topcon found no cell in 30 % of guttae images; success 70 % (ECD/CV) and 40 % (HEX) on guttae images (verified-from-source).
- Bland-Altman: 96.5 % ECD, 96.6 % CV, 96.7 % HEX within limits; Kruskal-Wallis vs manual P = 0.81 (ECD), 0.74 (CV), < 0.001 (HEX) (verified-from-source).
- HEX vertex method MAE 3.14 vs neighbour method 4.13 (verified-from-source).
- Edge threshold 0.1 chosen because it minimised CV error (verified-from-source).

## Limitations
- Author-stated: HEX still biased; very low-cell (grade 6, 13 +/- 10 cells) images unreliable; data on request.
- Reviewer: no r or MAPE tabulated; labels are manual masks, not device values.

## Relevance to CD / CV / HEX
Vendor-vs-manual disagreement on clean images (CV MAE 4.19, HEX MAE 8.34 points) is large relative to typical CV ~0.30-0.45, which matches our finding that device labels follow their own definition (device CV about twice Voronoi CV). HEX depends on counting method (vertex vs neighbour, 1-point MAE difference) and is underestimated with < 100 cells; CLEAR-EC labels use about 150 cells, so HEX labels carry this bias too.

## What it suggests we try
Oversample hard strata in each batch (dark/vignetted 2023-24 frames, outer CV deciles) as this paper did for guttae; and when computing HEX features from detected centers, compute both a neighbour-count share (Delaunay) and a Voronoi-vertex share and let the regressor pick.

Update 2026-11-09 (coordinator root-cause analysis): our CV/HEX error already equals an expert-centre reader plus ~150-cell sampling; better segmentation and label-guided segmenter self-training are closed for CV and HEX. Keep the line above as background, or for CD only.

## Related notes
- [[viguerasguillen2020dsaek]] — predecessor with CNN-ROI
- [[viguerasguillen2019fcn]] — base U-net and postprocessing
- [[sierra2023sdm]] — alternative guttae handling via signed distance maps

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9388684
- https://arxiv.org/abs/2203.01882
