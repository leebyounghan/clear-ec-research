---
id: tyagi2025boxcell
type: paper
title: "BoxCell: Leveraging SAM for Cell Segmentation with Box Supervision"
authors: [Aayush Kumar Tyagi, Vaibhav Mishra, Prathosh AP, Mausam]
year: 2025
venue: "Scientific Reports"
doi: "10.1038/s41598-025-26789-1"
arxiv: "2311.17960"
pmcid: "PMC12663235"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC12663235/"
citation: "Tyagi AK, Mishra V, Prathosh AP, Mausam. BoxCell: Leveraging SAM for Cell Segmentation with Box Supervision. Sci Rep. 2025;15:42652. doi:10.1038/s41598-025-26789-1"
tier: candidate
group: t4-cell-foundation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "Weakly supervised cell segmentation: frozen SAM turns gold boxes into pseudo-masks that train a standalone segmenter; at test time a YOLO detector's boxes prompt SAM and an ILP reconciles both masks; +5-8 Dice over the best SAM box-supervised baseline on three H&E nuclei sets."
relevance_clearec: medium
relevance_metrics: [CV, HEX]
relevance_note: "Shows that a frozen SAM plus cheap location prompts can manufacture training masks; our point detector or the 385 Topcon points could play the role of the boxes, but it also shows detector errors propagate directly into masks."
code_release: open
weights_release: unknown
data_scale: "CoNSeP 98/10/56, MoNuSeg 133/15/56, TNBC 34/5/11 train/val/test sub-images (H&E)"
key_numbers:
  - "BoxCell-ILP Dice 81.39 / 81.74 / 85.01 on CoNSeP / MoNuSeg / TNBC vs best SAM baseline 74.64 / 73.58 / 79.64 (verified-from-source)"
  - "AJI 44.72 / 63.75 / 66.26 vs best baseline 34.87 / 56.91 / 56.87 (verified-from-source)"
  - "Cross-domain (CoNSeP->MoNuSeg) Dice falls to 57.87; with stain variation CoNSeP 73.80 vs 81.39 (verified-from-source)"
  - "ILP reconciliation about 67-71 s per image with Gurobi, sparse variant about 15 s (verified-from-source)"
results:
  - {task: "binary nuclei segmentation", dataset: "MoNuSeg", metric: "Dice", value: "81.74", comparator: "SAM-BBTP++", comparator_value: "73.58", locator: "Results table", provenance: "verified-from-source"}
suggests_try: "Prompt frozen Cellpose-SAM/SAM with our detector's cell centres (points, or small boxes sized from label CD) to make pseudo-masks on the 8953 frames, then train a mask model on only the frames where pseudo-mask CD/HEX agree with the image labels."
superseded: "2026-11-09: segmentation / self-training route closed for CV and HEX by the root-cause analysis; see body"
tags: [weak-supervision, SAM, box-prompts, pseudo-masks, nuclei-segmentation]
search_keywords: [BoxCell, box-supervised cell segmentation, SAM pseudo-masks, YOLO prompts SAM]
uses_dataset: [CoNSeP, MoNuSeg, TNBC]
uses_method: [SAM box prompting, CaraNet, YOLOv8, integer linear programming, DenseCRF]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Detector errors propagate (false boxes give false masks, missed cells never segmented); weaker on low-contrast images", locator: "Limitations"}
  - {stated_by: author, note: "No quality filtering of SAM pseudo-masks; small H&E datasets; binary segmentation only", locator: "Method / Limitations"}
  - {stated_by: reviewer, note: "Needs gold boxes at train time (we have none on our frames; only predicted points), and no fully supervised reference is reported", locator: "Results"}
---

# BoxCell: Leveraging SAM for Cell Segmentation with Box Supervision

## TL;DR
BoxCell uses a frozen SAM twice: at training time gold boxes prompt SAM to create pseudo-masks that train a CaraNet segmenter; at test time a YOLOv8x detector's boxes prompt SAM and an integer linear program merges SAM's and the segmenter's masks. It beats prior box-supervised methods by about 5-10 Dice on three small H&E nuclei datasets.

## Citation and link
- Tyagi AK, Mishra V, Prathosh AP, Mausam. Sci Rep. 2025;15:42652. doi:10.1038/s41598-025-26789-1 (arXiv 2311.17960)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12663235/

## Problem and motivation
Full masks are expensive; boxes are cheap. SAM converts boxes to masks but makes errors; the question is how to use SAM without fine-tuning to train a standalone cell segmenter.

## Data
H&E only. CoNSeP (98/10/56 500x500 sub-images), MoNuSeg (133/15/56), TNBC (34/5/11, 512x512).

## Method
Train: gold boxes -> SAM -> pseudo-masks (no filtering) -> CaraNet (Dice + BCE). Test: YOLOv8x boxes -> SAM mask; CaraNet mask; ILP with per-patch RGB GMM intensity terms and spatial constraints merges them; DenseCRF refinement.

## Results
- Dice BoxCell-ILP 81.39 / 81.74 / 85.01 vs best SAM baseline 74.64 / 73.58 / 79.64 (verified-from-source).
- AJI 44.72 / 63.75 / 66.26 vs 34.87 / 56.91 / 56.87 (verified-from-source).
- Cross-domain C->M Dice 57.87 (verified-from-source).
- Fully supervised reference: not reported.

## Limitations
- Author-stated: detector error propagation, low-contrast failure, slow ILP (about 67-71 s/image), small datasets, binary only.
- Reviewer: no pseudo-mask quality control, which in our case is exactly where the image labels could help.

## Relevance to CD / CV / HEX
Our detector misses 15 % of cells and adds 16.5 % false ones; a BoxCell-style pipeline would turn those errors straight into merged/split masks (bad for area CV and HEX). The value for us is the generic recipe "cheap prompts + frozen SAM -> pseudo-masks -> train segmenter", provided we filter pseudo-masks with the image-level labels we have.

## What it suggests we try
Point-prompt Cellpose-SAM/SAM with detector centres on the 8953 frames, keep only frames whose pseudo-mask CD (cells/mm2 in the analysed region) is within ~5-10 % of the label CD and whose HEX is within a few points of label HEX, and train a mask model on those.

Update 2026-11-09 (coordinator root-cause analysis): our CV/HEX error already equals an expert-centre reader plus ~150-cell sampling; better segmentation and label-guided segmenter self-training are closed for CV and HEX. Keep the line above as background, or for CD only.

## Related notes
- [[pachitariu2025cellposesam]] — the frozen backbone we would prompt
- [[marks2025cellsam]] — automatic box prompts from a detector, same idea
- [[reith2025selfadapt]] — iterative teacher-student alternative

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12663235/
