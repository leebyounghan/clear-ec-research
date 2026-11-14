---
id: joseph2020postkeratoplasty
type: paper
title: "Quantitative and qualitative evaluation of deep learning automatic segmentations of corneal endothelial cell images of reduced image quality obtained following cornea transplant"
authors: [Naomi Joseph, Chaitanya Kolluru, Beth A. M. Benetz, Harry J. Menegay, Jonathan H. Lass, David L. Wilson]
year: 2020
venue: "Journal of Medical Imaging"
doi: "10.1117/1.JMI.7.1.014503"
arxiv: ""
pmcid: "PMC7019185"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7019185"
citation: "Joseph N, Kolluru C, Benetz BAM, Menegay HJ, Lass JH, Wilson DL. Quantitative and qualitative evaluation of deep learning automatic segmentations of corneal endothelial cell images of reduced image quality obtained following cornea transplant. J Med Imaging (Bellingham). 2020;7(1):014503. doi:10.1117/1.JMI.7.1.014503"
tier: candidate
group: t1-endothelium-segmentation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "Flattening + U-Net + Otsu-based postprocessing on 130 Konan post-DSAEK CPTS images reaches cell-wise Dice 0.87 and a cell count within ~4 % of readers inside the annotated region, and finds 564 extra cells outside it (about 70 % judged acceptable); ECD/CV/HEX are not validated."
relevance_clearec: medium
relevance_metrics: [CD]
relevance_note: "Shows that readers annotate only a sub-region and an automatic method finds many acceptable cells outside it, which mirrors CLEAR-EC labels coming from a technician-chosen ~17 % box."
code_release: unknown
weights_release: unknown
data_scale: "130 Konan specular images from CPTS post-DSAEK eyes (100 train/val, 30 test)"
key_numbers:
  - "Cell-wise Dice 0.87 +/- 0.17, Jaccard 0.80; 92 % of automated cells Dice > 0.80 (verified-from-source)"
  - "Inside reader regions: 1876 manual vs 1798 automated cells (about 4 % fewer) (verified-from-source)"
  - "564 new cells outside reader regions, about 70 % (390) judged acceptable (verified-from-source)"
  - "Experts flagged 52 of 1876 cells (3 %); 44 (2.3 %) judged better changed (verified-from-source)"
  - "ECD/CV/HEX agreement: not reported (verified-from-source)"
results:
  - {task: "cell segmentation", dataset: "30 held-out CPTS post-DSAEK images", metric: "cell-wise Dice", value: "0.87", comparator: "reader borders", comparator_value: "reference", locator: "Results", provenance: verified-from-source}
suggests_try: "Remember the label is a sub-region statistic: restrict any cell-statistic features to the 538 x 408 label box (or a predicted best box) instead of the full frame, since readers/technicians systematically pick a sub-area and cells outside it differ."
tags: [u-net, cpts, konan, post-keratoplasty, flattening, case-western]
search_keywords: [reduced image quality endothelial segmentation transplant, CPTS deep learning]
uses_dataset: [Cornea Preservation Time Study]
uses_method: [U-Net, image flattening, Otsu binarisation]
compares_with: []
extends: ["[[kolluru2019unet]]"]
has_limitation:
  - {stated_by: author, note: "ECD/CV/HEX not validated; single device (Konan) and transplant type; 30 test images", locator: "Discussion"}
  - {stated_by: reviewer, note: "Abstract mixes two different 92 % figures; 20 % vs 30 % new-cell figures inconsistent", locator: "Results/Fig. caption"}
---

# Deep learning segmentation of reduced-quality endothelial images after cornea transplant

## TL;DR
On 130 Konan images from the Cornea Preservation Time Study, division-based flattening plus U-Net and Otsu postprocessing gives cell-wise Dice 0.87 and reproduces reader cell counts within about 4 % inside the annotated region. The method also finds hundreds of acceptable cells outside the reader's region. Morphometrics are not validated.

## Citation and link
- Joseph N, et al. J Med Imaging. 2020;7(1):014503. doi:10.1117/1.JMI.7.1.014503
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7019185

## Problem and motivation
Post-keratoplasty images are low quality; reading-center analysis takes 15-30 min per image.

## Data
Konan specular microscopes; 130 CPTS images (DSAEK grafts, 6-48 months), 446 x 304 px; 8-130 cells per image, 600-2450 cells/mm2; reader borders via HAI CAS/EB corners. 100 train/val (10-fold), 30 test.

## Method
Flattening (division preferred over subtraction and Piorkowski-style localisation), 16-layer U-Net (93 x 93 receptive field), weighted BCE, Adam, 200 epochs, augmentation; adaptive Otsu, thinning, flood fill, area opening (< 50 px), pruning.

## Results
- Dice 0.87 +/- 0.17, Jaccard 0.80 +/- 0.18 (cell basis) (verified-from-source).
- 1798 automated vs 1876 manual cells inside reader regions (verified-from-source).
- 564 extra cells outside, ~70 % acceptable (verified-from-source).
- ECD/CV/HEX: not reported.

## Limitations
- Author-stated: morphometrics untested; single camera type; small test set.
- Reviewer: some cells have Dice 0.00 (complete failures), unquantified effect on CV.

## Relevance to CD / CV / HEX
The sub-region effect is the main lesson: the label region is chosen by a human and differs from the whole frame. In CLEAR-EC the technician box is about 17 % of the frame, so whole-frame features can disagree with the label.

## What it suggests we try
Compare CD/CV/HEX features computed inside the recorded box vs whole frame on OOF; if the box exists at test time, crop to it, otherwise train a box-attention or crop-ensemble that mimics technician choice (bright, well-focused area).

## Related notes
- [[kolluru2019unet]] — conference precursor
- [[daniel2019unet]] — Topcon real-world counterpart

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7019185
