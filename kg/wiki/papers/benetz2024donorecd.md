---
id: benetz2024donorecd
type: paper
title: "Automatic determination of endothelial cell density from donor cornea endothelial cell images"
authors: [Beth Ann M Benetz, Ved S Shivade, Naomi M Joseph, Nathan J Romig, John C McCormick, Jiawei Chen, et al.]
year: 2024
venue: "Translational Vision Science & Technology"
doi: "10.1167/tvst.13.8.40"
arxiv: ""
pmcid: "PMC11346145"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11346145"
citation: "Benetz BAM, Shivade VS, Joseph NM, Romig NJ, McCormick JC, Chen J, et al. Automatic determination of endothelial cell density from donor cornea endothelial cell images. Transl Vis Sci Technol. 2024;13(8):40. doi:10.1167/tvst.13.8.40"
tier: core
group: t2-device-morphometry
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "On US eye-bank donor images (Konan CellChek D), a self-supervised ViT/UNETR border segmenter estimated ECD with bias 123 cells/mm2 on the whole image but only 46 when restricted to a region matched to the area the eye bank counted (Lin's rc 0.77 vs 0.88); bias grew with ECD."
relevance_clearec: high
relevance_metrics: [CD]
relevance_note: "Closest published analogue of CLEAR-EC (US eye banks, Konan CellChek D, technician-counted region); shows that matching the technician's region roughly halves-to-thirds the CD bias."
code_release: none
weights_release: none
data_scale: "15,138 unlabelled images (8,169 donors) for SSL; 174 labelled; 3,024 test images, 1,939 with >= 100 cells by both methods"
key_numbers:
  - "Whole image: manual 2730 +- 349 vs automated 2607 +- 272 cells/mm2; bias 123 (95 % CI 114-131); Lin's rc 0.77; LoA -234 to 479; 81 % within 10 % (verified-from-source)"
  - "Cropped to region matched to the eye-bank counted area: automated 2683 +- 277; bias 46 (39-53); Lin's rc 0.88; LoA -243 to 335; 93 % within 10 % (verified-from-source)"
  - "Bias grew by about 134 (whole) and 120 (cropped) cells/mm2 per 500 cells/mm2 of ECD (verified-from-source)"
  - "Automated method found 100-1,263 cells per image vs 100-371 for manual counting (verified-from-source)"
  - "CV and HEX not reported (beyond scope) (verified-from-source)"
results:
  - {task: "donor ECD", dataset: "Eversight + VisionGift, Konan CellChek D", metric: "Lin's rc", value: "0.88 (cropped)", comparator: "whole image", comparator_value: "0.77", locator: "Results table", provenance: "verified-from-source"}
  - {task: "donor ECD", dataset: "Eversight + VisionGift, Konan CellChek D", metric: "bias cells/mm2", value: "46 (cropped)", comparator: "whole image", comparator_value: "123", locator: "Results table", provenance: "verified-from-source"}
suggests_try: "For CD, compute the point-based density only inside a predicted technician region (the densest, most legible window of about 17 % of the frame) and add a slope term that grows with density (their bias rose about 25 % of ECD per 500 cells/mm2), fitted on OOF MAPE."
tags: [eye-bank, konan, cellchek-d, donor-cornea, ssl, unetr, ecd, region-selection]
search_keywords: ["donor cornea endothelial cell density deep learning eye bank", "Konan CellChek D automated ECD", "Lass Wilson specular eye bank"]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "CV and HEX were beyond the study's scope.", locator: "Discussion"}
  - {stated_by: reviewer, note: "Manual eye-bank analysis procedure is not described beyond >= 40 (ideally > 100) cells; the cropped region was matched in area, not in location, to what technicians counted.", locator: "Methods"}
---

# Automatic determination of endothelial cell density from donor cornea endothelial cell images

## TL;DR
Case Western / Lass-Wilson group. Self-supervised ViT pretraining on 15k unlabelled eye-bank images, then a UNETR border segmenter trained on 174 images. ECD agreed with eye-bank manual counts much better when the automatic count was restricted to a region the size of the eye bank's counted area (bias 46 vs 123 cells/mm2), and bias rose with ECD.

## Citation and link
- Benetz BAM et al. TVST 2024;13(8):40. doi:10.1167/tvst.13.8.40
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11346145

## Problem and motivation
No fully automated donor-image ECD system is commercially available; eye banks rely on technicians with Konan software.

## Data
Konan CellChek D donor images from Eversight and VisionGift: 15,138 unlabelled images (8,169 donors), 174 labelled (81 + 93), 3,024 test images; 1,939 with >= 100 cells by both methods used for comparison.

## Method
ViT self-supervised pretraining, UNETR cell-border segmentation, post-processing removes stray borders, cells fully enclosed by borders are counted. Whole-image and cropped (circular region matched to the eye-bank counted area) analyses.

## Results
See key_numbers (verified-from-source). Donor age effect about -25 cells/mm2 per decade (whole image), judged not clinically significant.

## Limitations
- Author-stated: CV/HEX not studied.
- Reviewer: region matched by area only.

## Relevance to CD / CV / HEX
- CD: technician region choice matters: cells outside the counted region differ (brighter-region selection), and restricting to an equal-area region removed about 60 % of the bias. Our label box is about 17 % of the frame and the most-confident-peak window overlaps it (IoU 0.42).
- CV/HEX: not measured, but the region effect should be larger for CV/HEX than for CD.

## What it suggests we try
Compute CD (and emulated CV/HEX) inside a predicted technician window and use a density-dependent slope correction learned on OOF; do not average density over the whole frame.

## Related notes
- [[joseph2020postkeratoplasty]] — same group, earlier segmentation work.
- [[vidalvillegas2024eyebank]] — eye-bank Konan CellChek D reproducibility.
- [[huang2017autovsmanual]] — automatic Konan ECD overestimates in polymegethous corneas.

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11346145
