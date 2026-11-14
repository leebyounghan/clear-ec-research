---
id: pachitariu2025cellposesam
type: paper
title: "Cellpose-SAM: superhuman generalization for cellular segmentation"
authors: [Marius Pachitariu, Michael Rariden, Carsen Stringer]
year: 2025
venue: "bioRxiv"
doi: "10.1101/2025.04.28.651001"
arxiv: ""
pmcid: ""
url: "https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1.full"
citation: "Pachitariu M, Rariden M, Stringer C. Cellpose-SAM: superhuman generalization for cellular segmentation. bioRxiv. 2025. doi:10.1101/2025.04.28.651001"
tier: core
group: t4-cell-foundation
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [full-text, website]
evidence: preprint
tldr: "Replaces the Cellpose U-Net with a SAM ViT-L encoder (256x256 input, 8x8 patches, global attention, flow head) trained on 22,826 images / 3.3M ROIs from ~18-20 public datasets with heavy degradation and channel-permutation augmentation; test-set error 0.163 vs inter-annotator 0.257."
relevance_clearec: high
relevance_metrics: [CV, HEX, CD]
relevance_note: "Our zero-shot mask baseline and one of our regression backbones; the paper shows no endothelium/epithelium/specular data in training, which explains why zero-shot area CV only reaches r 0.38 and argues for domain adaptation rather than more zero-shot tuning."
code_release: open
weights_release: open
data_scale: "22,826 training images, 3,341,254 training ROIs; Cellpose, nuclei, TissueNet, LiveCell, Omnipose, YeaZ, DeepBacs, NeurIPS 2022, H&E nuclei sets, BCCD"
key_numbers:
  - "Cellpose test-set error rate: Cellpose-SAM 0.163 vs inter-annotator 0.257, estimated human-consensus bound 0.128, Cellpose3 0.292, CellSAM 0.328 (verified-from-source)"
  - "Training set: 22,826 images, 3,341,254 ROIs (verified-from-source)"
  - "ViT-L encoder (24 blocks, dim 1024, 305M of 312M parameters), input 256x256, patch 8x8 (verified-from-source)"
  - "Trained on cell diameters 7.5-120 px, no test-time resizing; small loss at small diameters (verified-from-source)"
  - "Robustness tested at Poisson noise scale 5/2.5/0.5, blur sigma 2/4/8, downsampling 2/5/10, anisotropic 2/6/12; matches best previous restoration models; per-condition AP only in figures (verified-from-source)"
  - "Training: 8 H200 GPUs, batch 256, about 20 h (2,000 epochs); fine-tuning fits 8-12 GB VRAM at batch 1 (verified-from-source)"
results:
  - {task: "instance segmentation", dataset: "Cellpose test set", metric: "error rate vs annotator 1", value: "0.163", comparator: "annotator 2 (inter-annotator)", comparator_value: "0.257", locator: "Results, Fig 2", provenance: "verified-from-source"}
  - {task: "instance segmentation", dataset: "Cellpose test set", metric: "error rate", value: "0.163", comparator: "CellSAM", comparator_value: "0.328", locator: "Results", provenance: "verified-from-source"}
suggests_try: "Treat Cellpose-SAM as a starting point to adapt on our 8953 frames (fine-tuning fits 8-12 GB at batch 1), not as a fixed zero-shot mask source; keep diameter at our native ~25-30 px (no rescale) and skip a separate restoration step, which the authors say Cellpose-SAM does not need."
superseded: "2026-11-09: segmentation / self-training route closed for CV and HEX by the root-cause analysis; see body"
tags: [foundation-model, SAM, cellpose, generalist-segmentation, robustness, flows]
search_keywords: [Cellpose-SAM, cpsam, Cellpose 4, SAM ViT-L cell segmentation, superhuman generalization]
uses_dataset: [Cellpose cyto2, TissueNet, LiveCell, Omnipose, NeurIPS 2022 cell segmentation, MoNuSeg, PanNuke]
uses_method: [SAM encoder, Cellpose flows, degradation augmentation, channel permutation]
compares_with:
  - {target: "[[stringer2025cellpose3]]", result: better, metric: "error rate on Cellpose test set", note: "0.163 vs 0.292", locator: "Results"}
  - {target: "[[marks2025cellsam]]", result: better, metric: "error rate on Cellpose test set", note: "0.163 vs 0.328", locator: "Results"}
extends: ["[[stringer2021cellpose]]"]
has_limitation:
  - {stated_by: author, note: "Small performance loss at small cell diameters", locator: "Results, robustness"}
  - {stated_by: reviewer, note: "No epithelial/endothelial packed monolayers or specular/reflective-light microscopy in training; the dataset count is given as 18 in Methods and 20 in Fig S1", locator: "Methods; Fig S1"}
  - {stated_by: reviewer, note: "Preprint, not peer-reviewed; robustness reported as figures, not per-condition numbers in text", locator: "whole paper"}
---

# Cellpose-SAM: superhuman generalization for cellular segmentation

## TL;DR
Cellpose-SAM keeps the Cellpose flow representation but swaps the CNN for a modified SAM ViT-L encoder, trained on 22,826 images from many public datasets with strong augmentation (degradations, channel permutation, size jitter). It roughly halves the error of Cellpose3 on the Cellpose test set and beats inter-annotator agreement. No corneal, endothelial or packed-epithelium data is in its training set.

## Citation and link
- Pachitariu M, Rariden M, Stringer C. bioRxiv 2025. doi:10.1101/2025.04.28.651001
- https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1.full ; code https://github.com/mouseland/cellpose

## Problem and motivation
Existing generalist segmenters reach roughly inter-annotator agreement but fail out of distribution; the authors argue the human-consensus bound (half the inter-annotator error) is the real ceiling and use a large pretrained backbone to approach it.

## Data
22,826 training images, 3,341,254 ROIs: Cellpose (cyto2, sampled 59 %), Cellpose nuclei (20 %), TissueNet (2,601 train / 1,249 test, fluorescence, 6 tissue types), LiveCell, Omnipose, YeaZ, DeepBacs, NeurIPS 2022, H&E nuclei (MoNuSeg, MoNuSAC, CryoNuSeg, NuInsSeg, CPM, TNBC, LynSec, IHC TMA, CoNIC, PanNuke), BCCD. Methods say 18 datasets, Fig S1 caption says 20. Epithelium, endothelium, cornea and monolayer are never mentioned.

## Method
- SAM ViT-L encoder initialised from SA-1B weights; input 1024 -> 256, patch 16 -> 8, global attention in all 24 blocks, SAM decoders removed; a transposed convolution outputs cell probability + horizontal/vertical flows (Cellpose post-processing).
- Augmentation: random channel permutation; diameters 7.5-120 px (scale 0.25-4 around 30 px); Poisson noise, blur, downsampling, anisotropic blur on 50 % of batch images.
- Trained on 8 H200 GPUs, batch 256, about 20 h. Fine-tuning possible at batch 1 on 8-12 GB.

## Results
- Error rate vs annotator 1 on the Cellpose test set: Cellpose-SAM 0.163; annotator 2 0.257; consensus estimate 0.128; Cellpose3 0.292; CellSAM 0.328 (verified-from-source).
- Channel permutation: "no performance loss"; restoration step "not required" for Cellpose-SAM (verified-from-source).
- Fastest model per image on A100 despite ~50x more parameters than Cellpose; exact seconds in Table S1, not read (verified-from-source for the claim; seconds not reported in text).
- Fine-tuning curves (1-512 images) only in Fig 4; AP gains not reported in text.

## Limitations
- Author-stated: small loss at small diameters.
- Reviewer: training domain excludes reflective-light specular microscopy of a packed hexagonal monolayer, where boundaries are dark lines and cell interiors are bright and textured; "superhuman" is measured against one annotator on fluorescence-type test sets, not on our domain.

## Relevance to CD / CV / HEX
Zero-shot Cellpose-SAM / watershed masks give area CV r 0.38 with our labels, much below expert markers (r 0.89). This paper shows why that is not surprising: none of its 3.3M ROIs come from our domain. Its robustness to noise/blur/channel order suggests the gap is a shape-prior/domain problem (packed polygonal cells, dark borders), not an image-quality problem. Our cells (CD roughly 2000-3000 /mm2 at 0.7716 um/px, our estimate about 25-30 px diameter) sit inside its trained diameter range.

## What it suggests we try
Use Cellpose-SAM as the student/teacher initialisation for label-free adaptation on the 8953 frames (see [[reith2025selfadapt]]), select checkpoints by mask-derived CV/HEX correlation with labels; run it at native scale with diameter near 27 px and no restoration pre-step.

Update 2026-11-09 (coordinator root-cause analysis): our CV/HEX error already equals an expert-centre reader plus ~150-cell sampling; better segmentation and label-guided segmenter self-training are closed for CV and HEX. Keep the line above as background, or for CD only.

## Related notes
- [[stringer2021cellpose]] — flow representation it inherits
- [[stringer2025cellpose3]] — restoration models it claims to make unnecessary
- [[marks2025cellsam]] — competing SAM-based generalist, worse here
- [[archit2025microsam]] — alternative SAM adaptation for microscopy
- [[reith2025selfadapt]] — label-free adaptation recipe for Cellpose models

## Sources
- https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1.full
- https://github.com/mouseland/cellpose (citation guidance via search snippet)
