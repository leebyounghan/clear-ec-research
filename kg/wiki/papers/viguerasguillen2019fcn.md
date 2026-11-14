---
id: viguerasguillen2019fcn
type: paper
title: "Fully convolutional architecture vs sliding-window CNN for corneal endothelium cell segmentation"
authors: [Juan P. Vigueras-Guillen, Busra Sari, Stanley F. Goes, Hans G. Lemij, Jeroen van Rooij, Koenraad A. Vermeer, et al.]
year: 2019
venue: "BMC Biomedical Engineering"
doi: "10.1186/s42490-019-0003-2"
arxiv: ""
pmcid: "PMC7412678"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7412678"
citation: "Vigueras-Guillen JP, Sari B, Goes SF, Lemij HG, van Rooij J, Vermeer KA, van Vliet LJ. Fully convolutional architecture vs sliding-window CNN for corneal endothelium cell segmentation. BMC Biomed Eng. 2019;1:4. doi:10.1186/s42490-019-0003-2"
tier: core
group: t1-endothelium-segmentation
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "U-net with Gaussian-blurred (probabilistic) edge labels plus Fourier-scale watershed segments 50 Topcon SP-1P images with 99.5 % correct cells; CV relative error 2.8 % and HEX 1.3 % vs manual, while the Topcon software is off by 10.1 % (CV) and 8.0 % (HEX) on the same images."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Gives a rare dose-response point: about 0.5-0.9 % wrongly split/merged cells already cost 2.8-3.6 % relative CV error; also shows Topcon CV/HEX differ from manual by 8-10 % even on clean images."
code_release: unknown
weights_release: unknown
data_scale: "50 Topcon SP-1P images (50 glaucomatous eyes), ~240 cells each, manual edge delineation"
key_numbers:
  - "U-net after postprocessing: DICE 0.981, MHD 0.22, over-segmented 0.220 %, under-segmented 0.260 %, correct cells 99.52 % (verified-from-source)"
  - "Relative error vs manual, U-net: ECD 0.4 %, CV 2.8 %, HEX 1.3 % (verified-from-source)"
  - "Relative error vs manual, SW-net (0.54 % over / 0.38 % under-segmented): ECD 0.8 %, CV 3.6 %, HEX 2.1 % (verified-from-source)"
  - "Topcon software vs manual: ECD 1.3 %, CV 10.1 %, HEX 8.0 % relative error; signed CV bias +1.9 points, HEX -2.2 points (verified-from-source)"
  - "Dataset ranges: ECD 1100-2800 cells/mm2, CV 18-36 %, HEX 44-74 %; pixel 1.038 um (verified-from-source)"
results:
  - {task: "CV estimation", dataset: "50 Topcon SP-1P images", metric: "relative error %", value: "2.8 (U-net)", comparator: "Topcon software", comparator_value: "10.1", locator: "Table 5", provenance: verified-from-source}
  - {task: "HEX estimation", dataset: "50 Topcon SP-1P images", metric: "relative error %", value: "1.3 (U-net)", comparator: "Topcon software", comparator_value: "8.0", locator: "Table 5", provenance: verified-from-source}
suggests_try: "Use the dose-response here (under 1 % wrong cells -> ~3 % relative CV error) to set expectations: our detector (15 % misses, 16.5 % false) is an order of magnitude worse, so mask-derived area CV cannot be trusted; train the edge/center model with Gaussian-blurred soft targets and pick watershed smoothing from the Fourier cell-size peak."
tags: [u-net, sliding-window, probabilistic-labels, fourier, watershed, topcon]
search_keywords: [U-net corneal endothelium, sliding window CNN endothelium, probabilistic edge labels]
uses_dataset: [Rotterdam glaucoma-eye specular images]
uses_method: [U-net, CLAHE, Fourier cell-size estimation, watershed]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Small set (50 images) of fairly clean glaucomatous corneas, single expert", locator: "Methods"}
  - {stated_by: reviewer, note: "No formal analysis of error propagation; dose-response must be read across the two networks", locator: "Table 4-5"}
---

# Fully convolutional architecture vs sliding-window CNN for corneal endothelium cell segmentation

## TL;DR
A U-net and a 64 x 64 sliding-window CNN are compared for edge detection on Topcon SP-1P images. Probabilistic (Gaussian-blurred) edge labels and a Fourier-scaled watershed turn the edge map into cells. U-net wins: 99.5 % correctly segmented cells and CV/HEX within 2.8 %/1.3 % of manual, far better than Topcon's own software (10.1 %/8.0 %).

## Citation and link
- Vigueras-Guillen JP, et al. BMC Biomed Eng. 2019;1:4. doi:10.1186/s42490-019-0003-2
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7412678

## Problem and motivation
Vendor software underestimates or fails on CV/HEX; the authors want reliable cell edges for clinical biomarkers.

## Data
Topcon SP-1P, 50 central-cornea images from 50 glaucomatous eyes; 240 x 528 px (0.25 x 0.55 mm, 1.038 um/px); ~240 cells per image; manual edges by one expert. ECD 1100-2800, CV 18-36 %, HEX 44-74 %.

## Method
- U-net (4 levels, 4 x 4 kernels, dropout 0.5) on raw images vs SW-net (64 x 64 patches, CLAHE).
- Edge labels convolved with a 7 x 7 Gaussian (SD 1 px) to make soft targets.
- Postprocessing: mean cell size from the Fourier ring peak, Gaussian smoothing of the CNN output with sigma tied to that size, then watershed.
- CD/CV/HEX computed from the segmentation inside the gold-standard area.

## Results
- AUC: U-net 0.9938, SW-net 0.9921 (verified-from-source).
- After postprocessing U-net DICE 0.981, MHD 0.22; over-/under-segmented cells 0.220 %/0.260 % (SW-net 0.537 %/0.382 %) (verified-from-source).
- Relative error vs manual (U-net / SW-net / Topcon): ECD 0.4/0.8/1.3 %, CV 2.8/3.6/10.1 %, HEX 1.3/2.1/8.0 % (verified-from-source).
- MAE (U-net / Topcon): CV 0.6 / 2.3 points, HEX 0.9 / 5.3 points (verified-from-source).
- Authors: "subtle differences had a significant effect in the biomarkers estimation" (verified-from-source).

## Limitations
- Author-stated: small, single-expert set.
- Reviewer: clean images with ~240 cells; does not cover guttae, darker frames or device-label targets.

## Relevance to CD / CV / HEX
CV is the most fragile metric: halving wrong-cell rate (0.92 % to 0.48 %) cut CV relative error from 3.6 % to 2.8 %, while ECD error was already below 1 %. This supports our finding that detector misses ruin area CV and that CV is the metric with the largest gap. It also shows the device's own CV differs from manual area CV, consistent with our "device CV about twice Voronoi CV" observation (the device follows its own definition and cell selection).

## What it suggests we try
Do not compute area CV from our current detector; if we use masks at all, use them only for neighbour/shape statistics that tolerate merges. For any edge/center model trained on the Topcon point set, use soft Gaussian targets and Fourier-peak-scaled smoothing before watershed.

## Related notes
- [[viguerasguillen2020dsaek]] — same pipeline extended with ROI detection
- [[piorkowski2017segmentation]] — how the segmentation method changes CV/HEX
- [[viguerasguillen2019regression]] — direct regression alternative

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7412678
