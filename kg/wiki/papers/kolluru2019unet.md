---
id: kolluru2019unet
type: paper
title: "Machine learning for segmenting cells in corneal endothelium images"
authors: [Chaitanya Kolluru, Beth A. Benetz, Naomi Joseph, Harry J. Menegay, Jonathan H. Lass, David Wilson]
year: 2019
venue: "Proceedings of SPIE Medical Imaging 2019"
doi: "10.1117/12.2513580"
arxiv: ""
pmcid: "PMC6874224"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6874224"
citation: "Kolluru C, Benetz BA, Joseph N, Menegay HJ, Lass JH, Wilson D. Machine learning for segmenting cells in corneal endothelium images. Proc SPIE Int Soc Opt Eng. 2019;10950:109504G. doi:10.1117/12.2513580"
tier: candidate
group: t1-endothelium-segmentation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "U-Net beats SegNet for border segmentation of 130 post-DSAEK CPTS specular images (Dice 0.86 vs 0.78 with adaptive thresholding, ~1 s/image), but ECD/CV/HEX agreement is left for future work."
relevance_clearec: low
relevance_metrics: [CD, CV, HEX]
relevance_note: "Background: early Case Western U-Net pipeline on reading-center (Konan/CPTS) images; shows illumination flattening and adaptive binarisation matter, no morphometric validation."
code_release: unknown
weights_release: unknown
data_scale: "130 CPTS post-DSAEK specular images (100 train/val, 30 test), single reader borders"
key_numbers:
  - "Held-out Dice: U-Net 0.86 (adaptive) / 0.83 (fixed); SegNet 0.78 / 0.59 (verified-from-source)"
  - "Images 446 x 304 px, pixel area 0.65 um2 (verified-from-source)"
  - "About 1 s per image end-to-end (verified-from-source)"
  - "ECD/CV/HEX agreement: not reported (verified-from-source)"
results:
  - {task: "border segmentation", dataset: "30 held-out CPTS images", metric: "Dice", value: "0.86 (U-Net adaptive)", comparator: "SegNet adaptive", comparator_value: "0.78", locator: "Table 1", provenance: verified-from-source}
suggests_try: "Nothing actionable beyond background; it only reinforces applying illumination flattening (row/column normalisation) before any classical cell statistic on our dark, vignetted 2023-24 frames."
tags: [u-net, segnet, cpts, konan, post-keratoplasty, case-western]
search_keywords: [Cornea Preservation Time Study segmentation, U-Net SegNet endothelium]
uses_dataset: [Cornea Preservation Time Study]
uses_method: [U-Net, SegNet, Sauvola thresholding]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Morphometrics not yet validated against HAI CAS/EB software", locator: "Discussion"}
  - {stated_by: reviewer, note: "Single reader, 30 test images, metrics only inside analyst region", locator: "Methods"}
---

# Machine learning for segmenting cells in corneal endothelium images

## TL;DR
The Case Western group compares U-Net and SegNet for cell-border segmentation of post-DSAEK specular images from the Cornea Preservation Time Study. U-Net with adaptive binarisation gets Dice 0.86 on 30 held-out images; clinical morphometrics are not evaluated.

## Citation and link
- Kolluru C, et al. Proc SPIE. 2019;10950:109504G. doi:10.1117/12.2513580
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6874224

## Problem and motivation
Reading-center analysis of endothelial images (HAI CAS/EB corner method) is slow; automated borders could speed it up.

## Data
130 CPTS post-DSAEK images (446 x 304 px), borders by one trained reader; 100 for ten-fold CV, 30 held out.

## Method
Gaussian or Piorkowski row/column flattening; U-Net (~7 M params) vs SegNet (~30 M); inverse-frequency weighted BCE; fixed vs Sauvola-like adaptive binarisation; closing, thinning, flood fill, small-component removal; ensemble of 3 best folds.

## Results
- Dice U-Net 0.86 / SegNet 0.78 (adaptive), 0.83 / 0.59 (fixed) (verified-from-source).
- ECD/CV/HEX: not reported (verified-from-source).

## Limitations
- Author-stated: no morphometric validation; small cohort.
- Reviewer: Konan reading-center images, not Topcon device output.

## Relevance to CD / CV / HEX
Only indirect: confirms flattening and local thresholds are needed on unevenly lit specular images, which applies to our vignetted 2023-24 frames.

## What it suggests we try
Nothing actionable; background only.

## Related notes
- [[joseph2020postkeratoplasty]] — journal follow-up on the same 130 images
- [[daniel2019unet]] — U-Net on Topcon real-world images

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6874224
