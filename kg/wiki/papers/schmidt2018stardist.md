---
id: schmidt2018stardist
type: paper
title: "Cell Detection with Star-convex Polygons"
authors: [Uwe Schmidt, Martin Weigert, Coleman Broaddus, Gene Myers]
year: 2018
venue: "MICCAI 2018 (LNCS 11071)"
doi: "10.1007/978-3-030-00934-2_30"
arxiv: "1806.03535"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/1806.03535"
citation: "Schmidt U, Weigert M, Broaddus C, Myers G. Cell Detection with Star-convex Polygons. MICCAI 2018, LNCS 11071:265-273. doi:10.1007/978-3-030-00934-2_30"
tier: candidate
group: t4-cell-foundation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html]
evidence: peer-reviewed
tldr: "StarDist predicts, per pixel, an object probability (normalised distance to background) and 32 radial distances to the boundary, then keeps polygons by NMS; it avoids merges of touching cells and beat U-Net and Mask R-CNN on DSB2018 (AP@0.5 0.864)."
relevance_clearec: medium
relevance_metrics: [CV, HEX, CD]
relevance_note: "Endothelial cells are convex polygons, a perfect fit for star-convex shapes; a StarDist head trained on pseudo-masks would output one polygon per cell (area for CV, count for CD) and suppress the merges that ruin area CV."
code_release: open
weights_release: open
data_scale: "Toy 1,000 synthetic, TRAgen 200 synthetic, DSB2018 497 fluorescence images; 90/10 splits"
key_numbers:
  - "DSB2018 AP@0.5: StarDist 0.864, Mask R-CNN 0.832, U-Net 3-class 0.806, U-Net 2-class 0.674 (verified-from-source)"
  - "DSB2018 AP@0.7: StarDist 0.685, Mask R-CNN 0.684, U-Net 3-class 0.654 (verified-from-source)"
  - "32 radial directions; weaker than baselines at IoU >= 0.85 on synthetic data (verified-from-source)"
  - "Runtime: not reported; about an order of magnitude fewer parameters than Mask R-CNN (about 45 M) (verified-from-source)"
results:
  - {task: "nuclei instance detection", dataset: "DSB2018 (497)", metric: "AP@0.5", value: "0.864", comparator: "Mask R-CNN", comparator_value: "0.832", locator: "Table 1", provenance: "verified-from-source"}
suggests_try: "If we self-train a mask model on filtered pseudo-masks from the 8953 frames, use a StarDist-style polygon head (object probability + radial distances) so each cell is one convex polygon; its centre probability also replaces our point detector, and polygon areas give area CV directly."
superseded: "2026-11-09: segmentation / self-training route closed for CV and HEX by the root-cause analysis; see body"
tags: [instance-segmentation, star-convex, polygons, nuclei, touching-cells]
search_keywords: [StarDist, star-convex polygons, radial distance cell segmentation]
uses_dataset: [DSB2018]
uses_method: [U-Net, radial distance regression, polygon NMS]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Parametric 32-ray shape limits accuracy at very high IoU", locator: "Results"}
  - {stated_by: reviewer, note: "Needs instance masks to train; nuclei are separated by background, whereas endothelial cells share borders with no gap", locator: "n/a"}
---

# Cell Detection with Star-convex Polygons

## TL;DR
StarDist represents each cell as a star-convex polygon: a U-Net predicts per-pixel object probability and distances to the boundary along 32 rays, and polygon NMS keeps one shape per cell. On crowded nuclei it avoids the merges of pixel-wise U-Nets and the box artefacts of Mask R-CNN.

## Citation and link
- Schmidt U, Weigert M, Broaddus C, Myers G. MICCAI 2018, LNCS 11071:265-273. doi:10.1007/978-3-030-00934-2_30
- https://ar5iv.labs.arxiv.org/html/1806.03535

## Problem and motivation
Touching cells get merged by semantic segmentation and suppressed by box-based detectors.

## Data
Toy synthetic ellipses (1,000), TRAgen synthetic crowded cells (200), DSB2018 fluorescence nuclei (497 of 670 images).

## Method
U-Net + 128-channel 3x3 conv, sigmoid probability head (target = normalised distance to background), linear 32-distance head; BCE + probability-weighted MAE; greedy polygon NMS.

## Results
- DSB2018 AP@0.5 0.864 vs Mask R-CNN 0.832 vs U-Net 0.806/0.674 (verified-from-source).
- AP@0.7 0.685 vs 0.684 vs 0.654 (verified-from-source).

## Limitations
- Author-stated: lower accuracy at very high IoU.
- Reviewer: requires masks; trained on fluorescence nuclei.

## Relevance to CD / CV / HEX
Corneal endothelial cells are near-hexagonal convex polygons, so the star-convex prior matches the shape class better than for most nuclei. Our point detector misses 15 % of cells (merges) and adds 16.5 % false ones; a polygon head regularises both by forcing one convex shape per centre and gives areas directly for CV and counts for CD.

## What it suggests we try
Train a StarDist head on pseudo-masks (Voronoi or Cellpose-SAM masks kept only when their CD/HEX agree with the image labels) and compare its area CV with the zero-shot mask r 0.38 on held-out folds.

Update 2026-11-09 (coordinator root-cause analysis): our CV/HEX error already equals an expert-centre reader plus ~150-cell sampling; better segmentation and label-guided segmenter self-training are closed for CV and HEX. Keep the line above as background, or for CD only.

## Related notes
- [[pachitariu2025cellposesam]] — flow-based alternative
- [[qu2020pointnuclei]] — Voronoi pseudo-labels from points to train the head
- [[archit2025microsam]] — distance-map decoder with seeded watershed

## Sources
- https://ar5iv.labs.arxiv.org/html/1806.03535
