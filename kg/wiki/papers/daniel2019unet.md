---
id: daniel2019unet
type: paper
title: "Automated segmentation of the corneal endothelium in a large set of 'real-world' specular microscopy images using the U-Net architecture"
authors: [Moritz C. Daniel, Lisa Atzrodt, Felicitas Bucher, Katrin Wacker, Stefan Bohringer, Thomas Reinhard, et al.]
year: 2019
venue: "Scientific Reports"
doi: "10.1038/s41598-019-41034-2"
arxiv: ""
pmcid: "PMC6426887"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6426887/"
citation: "Daniel MC, Atzrodt L, Bucher F, Wacker K, Bohringer S, Reinhard T, Bohringer D. Automated segmentation of the corneal endothelium in a large set of 'real-world' specular microscopy images using the U-Net architecture. Sci Rep. 2019;9:4752. doi:10.1038/s41598-019-41034-2"
tier: core
group: t1-endothelium-segmentation
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "U-Net trained on watershed labels grown from expert centroid dots in 385 Topcon SP-3000 images finds only 34 % of cells (precision 0.84) yet estimates ECD with R2 0.96 vs manual, because ECD is taken as the inverse median Voronoi area of the detected cells."
relevance_clearec: high
relevance_metrics: [CD]
relevance_note: "This 385-image Topcon centroid set appears to be the external point set our detector was trained on; it shows density from the median Voronoi area of a confident subset is robust to very low recall."
code_release: unknown
weights_release: unknown
data_scale: "385 Topcon SP-3000 images from 385 patients (158 gradable train, 227 validation), expert centroid dots"
key_numbers:
  - "385 images, 260 x 476 px, digitised from analog video prints; median 90 dotted cells per image (verified-from-source)"
  - "U-Net recall 0.34, precision 0.84 on centroids (8 px tolerance) (verified-from-source)"
  - "ECD R2 vs manual 0.96 (U-Net) vs 0.35 (marker-driven watershed); 92 % of images within +/-250 cells/mm2, 99 % within +/-500 (verified-from-source)"
  - "61 images ungradable by experts; U-Net flagged 92 as ungradable (verified-from-source)"
  - "CV and HEX not reported (verified-from-source)"
results:
  - {task: "ECD estimation", dataset: "227 validation images, Topcon SP-3000", metric: "R2 vs manual", value: "0.96", comparator: "Vincent marker-driven watershed", comparator_value: "0.35", locator: "Results table", provenance: verified-from-source}
suggests_try: "For CD, replace count-based density with 1/median(Voronoi area) (or median nearest-neighbour spacing) over only high-confidence detections in the label box; this paper got R2 0.96 at recall 0.34, so our 15 % misses should barely matter for CD if we use robust local spacing rather than counts."
tags: [u-net, centroid-annotation, voronoi, topcon, real-world, guttae]
search_keywords: [real-world specular microscopy U-Net, centroid dots endothelial cells, inverse median Voronoi area density]
uses_dataset: [Freiburg Topcon SP-3000 385-image centroid set]
uses_method: [U-Net, dot-constrained watershed labels, Voronoi]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Low recall caused by removing objects under 1000 px; single rater per image; single center/device", locator: "Discussion"}
  - {stated_by: reviewer, note: "No CV/HEX evaluation; images are digitised analog prints, older than CLEAR-EC frames", locator: "Methods"}
---

# Automated segmentation of the corneal endothelium in 'real-world' specular images with U-Net

## TL;DR
Daniel et al. train a U-Net on 385 real-world Topcon SP-3000 images annotated only with cell-center dots (labels grown by dot-constrained watershed). Detection recall is low (0.34) but density from the inverse median Voronoi area of detected cells matches manual ECD with R2 0.96, far above a classical watershed (0.35). CV and HEX are not attempted.

## Citation and link
- Daniel MC, et al. Sci Rep. 2019;9:4752. doi:10.1038/s41598-019-41034-2
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6426887/

## Problem and motivation
Prior methods were tested on small, clean sets; clinical archives contain guttae, low density and ungradable frames.

## Data
- Topcon SP-3000, 385 images from 385 patients, 260 x 476 px, from analog video prints; healthy, diseased, graft follow-up and 61 ungradable images; 36 with multiple guttae; 92 below 1000 cells/mm2.
- Centroid dots by one of three experts per image; median 90 cells per image.
- 158 gradable images (11,376 patches of 78 x 78) for training, 227 for validation.

## Method
- U-Net (tf_unet, 3 layers, 26 root features), border vs background; labels from watershed seeded by the manual dots.
- Preprocessing: Gaussian blur, top-hat, histogram equalisation. Postprocessing: 60 % probability threshold, remove objects < 1000 px, centroids from distance-map maxima.
- ECD = 1 / median Voronoi cell area, excluding border centroids. Match tolerance 8 px.

## Results
- Mean detected cells per image 46 (U-Net) vs 95 (watershed) (verified-from-source).
- Recall 0.34, precision 0.84 (verified-from-source).
- ECD R2 0.96 vs 0.35; within +/-250 cells/mm2 for 92 % vs 35 % of images (verified-from-source).
- CV/HEX, device-software comparison: not reported.

## Limitations
- Author-stated: low recall from aggressive size filtering; single rater; single device/center.
- Reviewer: low recall would wreck area CV and HEX, which is presumably why they are not reported; the set is old analog prints at a different resolution from CLEAR-EC.

## Relevance to CD / CV / HEX
If this is our external Topcon point set, its labels are single-rater dots with a median of only 90 cells, and dots were not placed on ungradable regions; a detector trained on it may inherit low-recall behaviour on dark frames. The robust-density idea is directly usable for CD: ratio statistics (median area) survive misses, unlike area CV.

## What it suggests we try
Compute CD features as 1/median Voronoi area and from the median of k-nearest-neighbour distances over confident detections in the 538 x 408 box; compare to the CD head (OOF 6.19) and blend. For CV, do not use counts or raw areas from low-recall detections.

## Related notes
- [[viguerasguillen2019fcn]] — full-mask approach on Topcon SP-1P
- [[kolluru2019unet]] — Case Western U-Net work on Konan images

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6426887/
