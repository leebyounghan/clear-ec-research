---
id: bruch2025synthesis
type: paper
title: "Improving 3D deep learning segmentation with biophysically motivated cell synthesis"
authors: [Roman Bruch, Mario Vitacolonna, Elina Nurnberg, Simeon Sauer, Rudiger Rudolf, Markus Reischl]
year: 2025
venue: "Communications Biology"
doi: "10.1038/s42003-025-07469-2"
arxiv: "2408.16471"
pmcid: "PMC11724918"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11724918/"
citation: "Bruch R, Vitacolonna M, Nurnberg E, Sauer S, Rudolf R, Reischl M. Improving 3D deep learning segmentation with biophysically motivated cell synthesis. Commun Biol. 2025;8. doi:10.1038/s42003-025-07469-2"
tier: candidate
group: t8-synthetic
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "Cell geometries from a Cellular Potts simulation, rendered by an imaging simulation (PSF, brightness loss, noise) plus CycleGAN refinement, train a 3D StarDist that beats training on 37.5 h of manual annotation (SEG 0.534 vs 0.505); direct GAN label-to-image translation looks realistic but fails (label inconsistency)."
relevance_clearec: medium
relevance_metrics: [CV, HEX]
relevance_note: "Best recent sim-to-real recipe for tessellated tissue: simulate geometry with measured shape statistics, render with a physical imaging model, refine with CycleGAN; warns that pure GAN translation breaks label consistency."
code_release: open
weights_release: open
data_scale: "one HT-29 spheroid recording; 4 synthetic training volumes per method; 3 manually corrected test patches with 1001 nuclei"
key_numbers:
  - "SEG / DET: SimOptiGAN+ (biophysical placement) 0.534 / 0.800 vs manual-annotation training 0.505 / 0.785 vs pretrained Cellpose nuclei 0.332 / 0.512 (verified-from-source)"
  - "Upper bound (trained on test data) 0.593 / 0.891 (verified-from-source)"
  - "Direct unpaired label-to-image GAN (Mem2NucGAN-U) 0.103 / 0.155 despite lowest KID (verified-from-source)"
  - "Manual annotation cost 37.5 h vs 0.4-4.4 h for synthetic pipelines (verified-from-source)"
results:
  - {task: "3D nuclei instance segmentation", dataset: "HT-29 spheroid, 1001 annotated nuclei", metric: "SEG", value: "0.534 (SimOptiGAN+)", comparator: "StarDist trained on manual annotation", comparator_value: "0.505", locator: "Results table", provenance: verified-from-source}
suggests_try: "If we build synthetic endothelium for CV/HEX pretraining, render geometry with an explicit imaging model fit to our frames (Gaussian PSF/blur, dark cell borders, vignetting and brightness falloff like the 2023-24 images, noise) and use a GAN only as a light refinement; do not rely on unpaired label-to-image GAN alone, which can break the geometry the label depends on."
tags: [sim-to-real, cellular-potts, cyclegan, stardist, synthetic-data, label-consistency]
search_keywords: [biophysical cell synthesis segmentation, sim-to-real cell segmentation, CycleGAN synthetic microscopy labels]
uses_dataset: []
uses_method: [Cellular Potts Model, CycleGAN, StarDist, imaging simulation]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Single spheroid/cell line; manual work still needed for parameters and prototypes; performance may drop for different morphology or density", locator: "Discussion"}
  - {stated_by: reviewer, note: "3D fluorescence nuclei, not 2D specular endothelium; transfer of the recipe is by analogy", locator: "n/a"}
---

# Improving 3D deep learning segmentation with biophysically motivated cell synthesis

## TL;DR
Bruch et al. simulate tissue geometry with a Cellular Potts model whose shape statistics match a real spheroid, render nuclei through an imaging simulation and CycleGAN, and train StarDist only on these synthetic volumes. The result beats training on manual annotations. A GAN that maps labels straight to images scores best on realism (KID) but worst on segmentation, because images and labels stop matching.

## Citation and link
- Bruch R, et al. Commun Biol. 2025. doi:10.1038/s42003-025-07469-2 (arXiv:2408.16471)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11724918/ ; code: github.com/s-sauer/tissue-simulation-pipeline, github.com/bruchr/cell_synthesis

## Problem and motivation
3D annotation is too costly; can biophysically plausible synthetic data replace it?

## Data
One HT-29 spheroid recording; test ground truth: 3 manually corrected patches, 1001 nuclei (37.5 h).

## Method
- CompuCell3D Cellular Potts model; per-cell volume/surface/shape targets from real segmentation; parameters tuned by Wasserstein distance of feature distributions.
- Rendering: SimOptiGAN (prototype placement + depth brightness loss, PSF, downsampling, noise, CycleGAN optimisation); SimOptiGAN+ places nuclei inside simulated cells; Mem2NucGAN variants translate labels directly.
- 3D StarDist trained six times per method; SEG/DET metrics; Welch t-tests.

## Results
- SimOptiGAN+ 0.534/0.800 > manual 0.505/0.785 > SimOptiGAN 0.495/0.769 > Cellpose 0.332/0.512 (verified-from-source).
- Mem2NucGAN-U 0.103/0.155; Mem2NucGAN-P 0.044/0.061 (verified-from-source).

## Limitations
- Author-stated: single dataset; manual parameter work; morphology/density shift may hurt.
- Reviewer: different modality and dimension from CLEAR-EC.

## Relevance to CD / CV / HEX
For synthetic endothelium, the geometry is the label (area CV, six-neighbour share). This paper shows geometry must survive rendering: physical rendering plus light GAN refinement preserves it, pure GAN translation does not.

## What it suggests we try
Fit a simple renderer (dark Voronoi/curved borders, cell-interior texture, blur, vignetting, noise) to our frames by matching intensity histograms and power spectra, generate mosaics with sampled CV/HEX, and pretrain the CV/HEX head; verify on OOF whether pretraining widens CV prediction spread beyond 66 % of label spread.

## Related notes
- [[kucharski2023gan]] — mosaic + GAN for endothelium
- [[bucht2011simulation]] — specular image simulator with known morphometry

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11724918/
