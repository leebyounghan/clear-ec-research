---
id: marks2025cellsam
type: paper
title: "CellSAM: a foundation model for cell segmentation"
authors: [Markus Marks, Uriah Israel, Rohit Dilip, Qilin Li, Changhua Yu, Emily Laubscher, et al.]
year: 2025
venue: "Nature Methods"
doi: "10.1038/s41592-025-02879-w"
arxiv: "2311.11004"
pmcid: "PMC12695629"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC12695629/"
citation: "Marks M, Israel U, Dilip R, Li Q, Yu C, Laubscher E, et al. CellSAM: a foundation model for cell segmentation. Nat Methods. 2025;22(12):2585-2593. doi:10.1038/s41592-025-02879-w"
tier: candidate
group: t4-cell-foundation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "SAM made automatic for cells: an NMS-free Anchor-DETR detector (CellFinder) on SAM's ViT-B backbone proposes boxes that prompt SAM's mask decoder; trained on 10 datasets, it beats generalist Cellpose zero-shot but performance holds only near the training distribution."
relevance_clearec: low
relevance_metrics: [CV, HEX]
relevance_note: "Detector-then-SAM design matches our point-detector setup, but Cellpose-SAM reports CellSAM at about twice its error, and CellSAM runtime scales with cell count."
code_release: open
weights_release: open
data_scale: "10 datasets (TissueNet, DeepBacs, BriFiSeg, Cellpose, Omnipose, YeastNet, YeaZ, DSB 2018, H&E, internal Phase400); total image count not in main text"
key_numbers:
  - "LIVECell zero-shot F1 0.40 vs 0.13 for generalist Cellpose (verified-from-source)"
  - "CellSAM-human vs human-human agreement not significantly different: P = 0.18 tissue, 0.49 cell culture, 0.11 yeast, 0.90 bacteria (verified-from-source)"
  - "CellFinder uses 3,500 queries, NMS-free, chosen for densely packed cells (verified-from-source)"
  - "Under 1 s per image on GPU; about 12 s on CPU vs about 8 s for Cellpose (verified-from-source)"
results:
  - {task: "zero-shot instance segmentation", dataset: "LIVECell (held out)", metric: "F1", value: "0.40", comparator: "Cellpose generalist", comparator_value: "0.13", locator: "Results, zero-shot", provenance: "verified-from-source"}
suggests_try: "Nothing directly actionable beyond the idea of a dense NMS-free detector feeding a mask decoder; prefer Cellpose-SAM as base (lower reported error) and use our detector only as a prompt/QC source."
tags: [foundation-model, SAM, detector-prompting, DETR, generalist-segmentation]
search_keywords: [CellSAM, CellFinder, Anchor DETR cell detection, SAM box prompts cells]
uses_dataset: [TissueNet, LIVECell, Cellpose, Omnipose, DSB 2018, YeaZ, DeepBacs]
uses_method: [SAM, Anchor DETR, box prompting]
compares_with:
  - {target: "[[pachitariu2025cellposesam]]", result: worse, metric: "error rate on Cellpose test set (reported by Cellpose-SAM paper)", note: "0.328 vs 0.163", locator: "Cellpose-SAM Results"}
extends: []
has_limitation:
  - {stated_by: author, note: "Guarantees only near training distribution; fine-tuning could not recover SH-SY5Y; runtime linear in cell count", locator: "Discussion"}
  - {stated_by: reviewer, note: "No specular/endothelial data; boxes are a poor prompt for tightly tiled polygons where neighbouring boxes overlap heavily", locator: "n/a"}
---

# CellSAM: a foundation model for cell segmentation

## TL;DR
CellSAM adds a transformer detector (CellFinder, Anchor DETR on SAM's ViT-B) whose boxes prompt SAM, giving automatic cell segmentation across tissue, culture, H&E, yeast and bacteria. It beats generalist Cellpose on held-out data and matches human-human agreement on several categories, but generalisation is limited to images like its training data.

## Citation and link
- Marks M, Israel U, Dilip R, et al. Nat Methods. 2025;22(12):2585-2593. doi:10.1038/s41592-025-02879-w (preprint arXiv 2311.11004, Israel et al. 2023)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12695629/

## Problem and motivation
SAM needs prompts that encode where cells are; CellSAM learns to produce those prompts.

## Data
Ten datasets across brightfield, phase, fluorescence and mass cytometry; evaluation samples per category: tissue 330, culture 144, H&E 51, bacteria 260, yeast 32, nuclear 56. LIVECell held out. Epithelial monolayers not specifically mentioned.

## Method
Stage 1 trains ViT + CellFinder (2,800 epochs); stage 2 freezes ViT and mask decoder and tunes SAM's neck with ground-truth boxes (50 epochs). Box threshold 0.4 (k-means adjusted), IoU 0.5, pixel 0.5.

## Results
- LIVECell zero-shot F1 0.40 vs 0.13 for Cellpose (verified-from-source).
- Per-category error values only in Fig 2b (not reported in text).
- In the later Cellpose-SAM preprint CellSAM has error 0.328 vs 0.163 for Cellpose-SAM on the Cellpose test set (see [[pachitariu2025cellposesam]]).

## Limitations
- Author-stated: limited out-of-distribution guarantees, linear runtime in cell count, box prompts limit some morphologies.
- Reviewer: our frames hold several hundred tightly tiled cells; a box per cell and a SAM pass per cell is slow and boxes overlap neighbours.

## Relevance to CD / CV / HEX
Background for the detector-plus-mask design; relevant only if we prompt a mask model with detector output. Its own detector would face the same miss/false problems our point detector has (15 % / 16.5 %).

## What it suggests we try
Nothing actionable beyond using detector output as prompts or QC for a stronger mask model; background only.

## Related notes
- [[pachitariu2025cellposesam]] — stronger SAM-based generalist
- [[tyagi2025boxcell]] — box prompts to SAM for pseudo-masks
- [[archit2025microsam]] — alternative SAM fine-tuning for microscopy

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12695629/
