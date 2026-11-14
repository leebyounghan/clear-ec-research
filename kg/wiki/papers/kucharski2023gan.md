---
id: kucharski2023gan
type: paper
title: "Corneal endothelial image segmentation training data generation using GANs. Do experts need to annotate?"
authors: [Adrian Kucharski, Anna Fabijanska]
year: 2023
venue: "Biomedical Signal Processing and Control"
doi: "10.1016/j.bspc.2023.104985"
arxiv: ""
pmcid: ""
url: "https://api.openalex.org/works/doi:10.1016/j.bspc.2023.104985"
citation: "Kucharski A, Fabijanska A. Corneal endothelial image segmentation training data generation using GANs. Do experts need to annotate? Biomed Signal Process Control. 2023;85:104985. doi:10.1016/j.bspc.2023.104985"
tier: core
group: t8-synthetic
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "A mosaic generator draws cell-edge maps, a GAN renders them into realistic endothelium patches, and a patch U-Net trained only on these synthetic pairs segments real images about as well as one trained on expert masks (1-4 % lower depending on dataset)."
relevance_clearec: high
relevance_metrics: [CV, HEX]
relevance_note: "Training without manual masks: synthetic tessellations give exact cell geometry, so CV/HEX of the generated mosaic are known by construction; fits the no-annotation rule."
code_release: unknown
weights_release: unknown
data_scale: "three public endothelium datasets for evaluation; synthetic patches from a mosaic generator + GAN (counts not reported in abstract)"
key_numbers:
  - "Purely synthetic training: segmentation quality only 1 % to 4 % below expert-annotated training, depending on dataset (from-abstract)"
  - "Evaluated on three datasets (from-abstract)"
  - "Pipeline: grid-like binary edge mosaic -> GAN image translation -> patch-based U-Net (from-abstract)"
results:
  - {task: "endothelium segmentation", dataset: "three datasets (names not in abstract)", metric: "segmentation quality drop vs expert-label training", value: "1-4 %", comparator: "U-Net trained on expert masks", comparator_value: "reference", locator: "Abstract", provenance: from-abstract}
suggests_try: "Build a mosaic-to-image generator for our domain without labels: sample centroidal-Voronoi/perturbed-hexagonal mosaics with target CD/CV/HEX spanning the label range (CV 0.29-0.50), render them with an unpaired GAN or simple physics-style blur/noise/vignetting fit to our frames, and pretrain the CV/HEX head on exact synthetic targets before fine-tuning on device labels."
tags: [gan, synthetic-data, mosaic-generator, u-net, no-annotation]
search_keywords: [GAN synthetic corneal endothelium, training data generation endothelium, do experts need to annotate]
uses_dataset: []
uses_method: [GAN image-to-image translation, mosaic generator, patch U-Net]
compares_with: []
extends: ["[[kucharski2021cnnwatershed]]"]
has_limitation:
  - {stated_by: reviewer, note: "Only abstract available (Elsevier page blocked); dataset names, GAN type and CV/HEX effects unverified", locator: "Abstract"}
  - {stated_by: reviewer, note: "Evaluated on segmentation quality, not on CD/CV/HEX error; synthetic mosaic statistics may not match real polymegethism", locator: "Abstract"}
---

# Corneal endothelial segmentation training data generation using GANs

## TL;DR
Kucharski and Fabijanska generate synthetic training pairs: a custom mosaic generator draws grid-like cell-edge maps and a GAN turns each map into a realistic endothelium patch. A patch-based U-Net trained only on these pairs segments real images with quality just 1-4 % below expert-annotated training on three datasets.

## Citation and link
- Kucharski A, Fabijanska A. Biomed Signal Process Control. 2023;85:104985. doi:10.1016/j.bspc.2023.104985 (hybrid open access, CC BY-NC-ND per OpenAlex)
- Abstract read via OpenAlex inverted index; ScienceDirect and ResearchGate returned 403.

## Problem and motivation
CNN segmentation of endothelium is limited by the cost of tracing cell borders; can synthetic data replace experts?

## Data
Three endothelium datasets for evaluation (from earlier work by the same group, e.g. [[kucharski2021cnnwatershed]]); names and sizes not reported in what was read.

## Method
- Mosaic generator produces binary edge maps that imitate endothelial cell tessellations.
- GAN translates edge maps into synthetic image patches (paired by construction).
- Patch-based U-Net trained on synthetic pairs; compared to the same U-Net trained on expert annotations.
- A simple GUI is provided for generating data (from-abstract).

## Results
- Purely synthetic training is comparable to expert-label training; quality drop 1-4 % depending on dataset (from-abstract).
- CD/CV/HEX errors: not reported in what was read.

## Limitations
- Author-stated: not available beyond the abstract.
- Reviewer: GANs can hallucinate structure that does not match the mask; mosaic statistics (CV, hexagon share) set by the generator may not cover real polymegethism; no metric-level evaluation.

## Relevance to CD / CV / HEX
Gives a no-annotation path for a segmentation or point model on our device domain. More importantly for CLEAR-EC, synthetic mosaics give exact area CV and six-neighbour share, so a CV/HEX regressor could be pretrained on unlimited perfectly labelled images, addressing regression to the mean (our CV predictions span only 66 % of the label spread) by supplying many extreme-CV examples.

## What it suggests we try
Generate ~50-100k synthetic 538 x 408 patches from perturbed Voronoi mosaics with CV and HEX sampled uniformly over and beyond the label range, render with a learned (CycleGAN-style, unpaired with our real frames) or hand-tuned appearance model, pretrain the regression head on (CV_area, HEX) targets, then fine-tune on device labels with a learned linear map (device CV is about twice Voronoi CV).

## Related notes
- [[kucharski2021cnnwatershed]] — earlier marker-based segmentation by the same group
- [[fabijanska2018unet]] — U-Net endothelium segmentation, same lab
- [[nurzynska2024gan]] — another GAN synthetic-data study (Alizarine)

## Sources
- https://api.openalex.org/works/doi:10.1016/j.bspc.2023.104985
- https://api.crossref.org/works?query.bibliographic=... (bibliographic record)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13038023/ (review mention)
