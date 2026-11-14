---
id: stringer2025cellpose3
type: paper
title: "Cellpose3: one-click image restoration for improved cellular segmentation"
authors: [Carsen Stringer, Marius Pachitariu]
year: 2025
venue: "Nature Methods"
doi: "10.1038/s41592-025-02595-5"
arxiv: ""
pmcid: "PMC11903308"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11903308/"
citation: "Stringer C, Pachitariu M. Cellpose3: one-click image restoration for improved cellular segmentation. Nat Methods. 2025;22(3):592-599. doi:10.1038/s41592-025-02595-5"
tier: candidate
group: t4-cell-foundation
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "Trains denoise/deblur/upsample networks against a frozen Cellpose segmentation loss (plus a perceptual loss) so restored images segment well; also a super-generalist cyto3 model on 8,402 images. It does not model illumination non-uniformity or vignetting."
relevance_clearec: low
relevance_metrics: [CV, HEX]
relevance_note: "Answers our question 'does Cellpose3 restoration help darker vignetted 2023-24 frames': it targets Poisson noise, blur and undersampling only, and the later Cellpose-SAM paper says the restoration step is not required for Cellpose-SAM."
code_release: open
weights_release: open
data_scale: "Restoration: cyto2 796 train images, nuclei 1,025 train; super-generalist 8,402 images from nine datasets"
key_numbers:
  - "Super-generalist (cyto3) AP@0.5 0.786 on the Cellpose test set vs 0.771 in the dataset-specific comparison (verified-from-source)"
  - "Training degradations: Poisson noise (gamma alpha 4.0, beta 0.7), blur s.d. 1-10, downsampling x2-7, each on 80 % of images (verified-from-source)"
  - "Noise2Self baseline average AP@0.5 0.435 (default CNN) and 0.511 (study U-net) (verified-from-source)"
  - "Per-degradation before/after AP only in figures: not reported in text (verified-from-source)"
results:
  - {task: "instance segmentation", dataset: "Cellpose test set", metric: "AP@0.5", value: "0.786", comparator: "dataset-specific", comparator_value: "0.771", locator: "Results, super-generalist", provenance: "verified-from-source"}
suggests_try: "Do not spend time on Cellpose3 restoration for the dark vignetted 2023-24 frames; instead apply explicit flat-field correction (divide by a large-sigma blurred image) or train with vignetting/brightness augmentation, and check CD error per source (old 7.11 vs new 5.50)."
tags: [image-restoration, denoising, segmentation-aware-loss, cellpose, generalist-model]
search_keywords: [Cellpose3, cyto3, one-click restoration, segmentation loss restoration]
uses_dataset: [Cellpose cyto2, Cellpose nuclei, TissueNet, LiveCell]
uses_method: [restoration network, perceptual loss, frozen segmentation loss]
compares_with: []
extends: ["[[stringer2021cellpose]]"]
has_limitation:
  - {stated_by: author, note: "Possible hallucinated features; perceptual loss improves looks not segmentation; limited out-of-distribution generalisation for all methods", locator: "Discussion"}
  - {stated_by: reviewer, note: "No illumination non-uniformity/vignetting model; only per-image percentile normalisation", locator: "Methods"}
---

# Cellpose3: one-click image restoration for improved cellular segmentation

## TL;DR
Cellpose3 trains image restoration networks (denoise, deblur, upsample) whose loss is the segmentation error of a frozen Cellpose model plus a perceptual term, so restored images are optimised to be segmentable. It also releases a super-generalist cyto3 segmentation model. Degradations covered are Poisson noise, Gaussian blur and undersampling, not uneven illumination.

## Citation and link
- Stringer C, Pachitariu M. Nat Methods. 2025;22(3):592-599. doi:10.1038/s41592-025-02595-5
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11903308/

## Problem and motivation
Generalist segmenters fail on noisy, blurred or undersampled images; classical restoration optimises pixel fidelity, not segmentability.

## Data
cyto2 (796 training images), nuclei (1,025 training, 111 test); super-generalist sampled from 8,402 images across nine datasets. Test degradations tuned to roughly halve AP.

## Method
Restoration U-Net chained to frozen Cellpose; loss = segmentation loss (flows/probability vs ground truth) + perceptual loss on Cellpose activation correlations. AdamW, 2,000 epochs.

## Results
- cyto3 super-generalist AP@0.5 0.786 vs 0.771 (verified-from-source).
- Restoration gains for noisy/blurry/undersampled images shown in Figs 2-4; exact values not reported in text.
- Beats Noise2Void, Noise2Self and CARE out of the box; retrained blind methods may do better with many similar images (verified-from-source, qualitative).

## Limitations
- Author-stated: hallucination risk, perceptual loss does not help segmentation, users must supply diameter.
- Reviewer: specular endothelial images are not noise- or blur-limited in the usual sense; their problem is vignetting, low contrast at the edges and guttae/dark regions, which this method does not target.

## Relevance to CD / CV / HEX
The 2023-24 images are darker and vignetted and the old-format source has higher CD error (7.11 vs 5.50). Cellpose3 is not designed for this shift, and Cellpose-SAM ([[pachitariu2025cellposesam]]) already absorbs its degradations. The segmentation-loss-for-restoration idea is still notable: one could train a preprocessing net to maximise downstream metric accuracy.

## What it suggests we try
Skip Cellpose3 restoration; use flat-field correction plus brightness/vignetting augmentation and compare per-source CD/CV error before and after.

## Related notes
- [[pachitariu2025cellposesam]] — states restoration is not required for Cellpose-SAM
- [[stringer2021cellpose]] — base segmentation model

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11903308/
