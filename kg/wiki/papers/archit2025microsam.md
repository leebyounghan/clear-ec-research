---
id: archit2025microsam
type: paper
title: "Segment Anything for Microscopy"
authors: [Anwai Archit, Luca Freckmann, Sushmita Nair, Nabeel Khalid, Paul Hilt, Vikas Rajashekar, et al.]
year: 2025
venue: "Nature Methods"
doi: "10.1038/s41592-024-02580-4"
arxiv: ""
pmcid: "PMC11903314"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11903314/"
citation: "Archit A, Freckmann L, Nair S, Khalid N, Hilt P, Rajashekar V, et al. Segment Anything for Microscopy. Nat Methods. 2025;22(3):579-591. doi:10.1038/s41592-024-02580-4"
tier: candidate
group: t4-cell-foundation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "micro-sam fine-tunes SAM for light and electron microscopy, adds an automatic instance segmentation (AIS) decoder predicting centre/boundary distances + foreground with seeded watershed, and a napari tool for interactive annotation and fine-tuning; automatic quality is on par with Cellpose on LIVECell."
relevance_clearec: low
relevance_metrics: [CV, HEX]
relevance_note: "Alternative SAM-based base model; its gains come from annotation-based fine-tuning and interactive correction, which our no-manual-annotation rule forbids; no label-free adaptation is discussed."
code_release: open
weights_release: open
data_scale: "LM generalist: LIVECell, DeepBacs, TissueNet, NeurIPS CellSeg, PlantSeg root, DSB, StarDist set, 8 Cell Tracking Challenge datasets (counts in Suppl. Table 1, not read)"
key_numbers:
  - "LIVECell decoder comparison (mSA): AIS distances 0.39, affinities 0.36, boundaries 0.31 (verified-from-source)"
  - "Generalist training: 250,000 iterations on one A100 80 GB, about 6 days (verified-from-source)"
  - "COVID-IF fine-tuning: 30 min on GPU, 5.3 h on CPU (verified-from-source)"
  - "Runtime: Cellpose about 0.3 s/image GPU; micro-sam embedding + AIS 0.2-1.2 s GPU depending on model size (verified-from-source)"
results:
  - {task: "automatic instance segmentation", dataset: "LIVECell", metric: "mSA", value: "0.39 (AIS)", comparator: "boundary decoder", comparator_value: "0.31", locator: "Results", provenance: "verified-from-source"}
suggests_try: "Nothing actionable under the no-annotation rule; if we want a second mask model for ensembling or teacher diversity, the micro-sam AIS decoder (centre/boundary distance + seeded watershed) is a reasonable alternative to Cellpose flows for packed cells."
tags: [SAM, microscopy, interactive-segmentation, napari, fine-tuning]
search_keywords: [micro-sam, muSAM, Segment Anything for Microscopy, AIS decoder]
uses_dataset: [LIVECell, TissueNet, NeurIPS CellSeg, DSB 2018, Cell Tracking Challenge]
uses_method: [SAM, UNETR decoder, seeded watershed]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Automatic segmentation only on par with Cellpose; does not claim superiority; poor automatic results on Lizard and mouse embryo", locator: "Results"}
  - {stated_by: reviewer, note: "Adaptation path assumes manual/interactive labels; no pseudo-label or label-free mode", locator: "whole paper"}
---

# Segment Anything for Microscopy

## TL;DR
micro-sam fine-tunes SAM on light- and electron-microscopy data, adds a UNETR decoder for automatic instance segmentation (distance maps + seeded watershed), and ships a napari plugin for interactive segmentation, tracking and user fine-tuning. Automatic results match Cellpose on LIVECell; the main gains are in interactive annotation.

## Citation and link
- Archit A, Freckmann L, Nair S, et al. Nat Methods. 2025;22(3):579-591. doi:10.1038/s41592-024-02580-4 (correction Nat Methods 2025;22(7):1603)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11903314/

## Problem and motivation
SAM is not trained on microscopy and needs prompts; micro-sam makes it microscopy-aware and usable for annotation.

## Data
LM generalist trained on LIVECell, DeepBacs, TissueNet, NeurIPS CellSeg, PlantSeg root, DSB, StarDist data (no histopathology), eight CTC datasets. Out-of-domain tests: COVID-IF, PlantSeg ovules, Lizard, mouse embryo.

## Method
SAM ViT-T/B/L/H encoders (reported generalist uses ViT-L); AIS decoder predicts centre distance, boundary distance and foreground, then seeded watershed.

## Results
- LIVECell mSA: AIS 0.39, affinities 0.36, boundaries 0.31 (verified-from-source).
- Fine-tuning gains mostly with 2-5 % of LIVECell training data; per-image counts not reported in text.
- Generalist beats default SAM on all out-of-domain sets (qualitative; numbers in figures).

## Limitations
- Author-stated: automatic segmentation not better than Cellpose; struggles on Lizard and mouse embryo.
- Reviewer: requires annotations to adapt; no endothelium data.

## Relevance to CD / CV / HEX
Background only: our constraint (no manual labels) removes its main advantage. Its distance-map decoder is a plausible second architecture if we build a teacher-student loop.

## What it suggests we try
Nothing actionable under the no-annotation rule; background only.

## Related notes
- [[pachitariu2025cellposesam]] — preferred base model
- [[marks2025cellsam]] — another SAM-for-cells approach

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11903314/
