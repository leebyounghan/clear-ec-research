---
id: viguerasguillen2020dsaek
type: paper
title: "Deep Learning for Assessing the Corneal Endothelium from Specular Microscopy Images up to 1 Year after Ultrathin-DSAEK Surgery"
authors: [Juan P. Vigueras-Guillen, Jeroen van Rooij, Angela Engel, Hans G. Lemij, Lucas J. van Vliet, Koenraad A. Vermeer]
year: 2020
venue: "Translational Vision Science & Technology"
doi: "10.1167/tvst.9.2.49"
arxiv: ""
pmcid: "PMC7445361"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7445361"
citation: "Vigueras-Guillen JP, van Rooij J, Engel A, Lemij HG, van Vliet LJ, Vermeer KA. Deep Learning for Assessing the Corneal Endothelium from Specular Microscopy Images up to 1 Year after Ultrathin-DSAEK Surgery. Transl Vis Sci Technol. 2020;9(2):49. doi:10.1167/tvst.9.2.49"
tier: core
group: t1-endothelium-segmentation
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "Dense U-Net edge network + ROI network + Fourier-smoothed watershed on Topcon SP-1P post-DSAEK images gives ECD/CV/HEX percentage errors of 2.5/5.7/5.7 % vs manual segmentation, against 7.5/17.5/18.3 % for the Topcon software."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Same device family (Topcon SP-1P) and same three metrics; shows the vendor software is far from manual truth on CV/HEX and that few segmentation errors distort HEX most."
code_release: partial
weights_release: unknown
data_scale: "383 post-UT-DSAEK Topcon SP-1P images (41 eyes) + 400 annotated Baerveldt-study images for training"
key_numbers:
  - "Percentage error vs manual: ECD 2.5 %, CV 5.7 %, HEX 5.7 % (Topcon software: 7.5 %, 17.5 %, 18.3 %) (verified-from-source)"
  - "Mean absolute error vs manual: ECD 30.0 cells/mm2, CV 1.5 points, HEX 3.0 points (Topcon: 118.7, 4.6, 10.3) (verified-from-source)"
  - "Estimate availability: ours 98.4 % (ECD/CV), Topcon 71.5 % ECD/CV and 30.5 % HEX (verified-from-source)"
  - "Cells per image: 115 (ours) vs 30 (Topcon) on average (verified-from-source)"
  - "CV slightly overestimated when fewer than 50 cells are detected; HEX needs more cells than ECD/CV to cut error spread (verified-from-source)"
results:
  - {task: "ECD/CV/HEX estimation", dataset: "383 post-UT-DSAEK Topcon SP-1P images", metric: "percentage error vs manual", value: "2.5 / 5.7 / 5.7", comparator: "Topcon IMAGEnet software", comparator_value: "7.5 / 17.5 / 18.3", locator: "Results / Abstract", provenance: verified-from-source}
suggests_try: "Treat area CV as fragile under segmentation errors (CV is overestimated below ~50 cells) and give the CV head inputs that do not depend on full masks; check our CV error vs detected-cell count in the label box the way this paper plots error spread vs cell count."
tags: [dense-unet, watershed, roi-detection, topcon, post-keratoplasty, fuchs]
search_keywords: [corneal endothelium deep learning DSAEK, CNN-Edge CNN-ROI, Topcon SP-1P CV HEX error]
uses_dataset: [Rotterdam Eye Hospital UT-DSAEK, Baerveldt study images]
uses_method: [Dense U-Net, watershed, Fourier cell-size estimation]
compares_with: []
extends: ["[[viguerasguillen2019fcn]]"]
has_limitation:
  - {stated_by: author, note: "Single center, 41 FED eyes; guttae-rich images still need improvement", locator: "Discussion"}
  - {stated_by: reviewer, note: "Ground truth is manual segmentation, not device output; the Topcon software is the comparator, not the target as in CLEAR-EC", locator: "Methods"}
---

# Deep Learning for Assessing the Corneal Endothelium after UT-DSAEK

## TL;DR
A three-stage pipeline (CNN-Edge, CNN-ROI, watershed postprocessing) segments post-transplant specular images and computes ECD, CV and HEX on the trustworthy region. Against manual segmentation it is three times more accurate than the Topcon built-in software, which also fails to give HEX on most images. The paper documents that HEX needs the most cells and that CV is biased upward when few cells are detected.

## Citation and link
- Vigueras-Guillen JP, et al. Transl Vis Sci Technol. 2020;9(2):49. doi:10.1167/tvst.9.2.49
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7445361 ; code notebook: https://www.github.com/jpviguerasguillen/corneal_endothelium_TVST

## Problem and motivation
Post-DSAEK images are low contrast and blurry; vendor software gives few cells and unreliable CV/HEX, so CV and HEX were not used clinically.

## Data
- Topcon SP-1P, IMAGEnet i-base 1.32. 383 images, 41 eyes, 1/3/6/12 months after UT-DSAEK for Fuchs dystrophy.
- Image 240 x 528 px, about 0.25 x 0.55 mm. Extra training set: 400 manually segmented Baerveldt-study images (non-diseased, ~2200 cells/mm2).
- Manual segmentation as gold standard; five-fold CV split by eye.

## Method
- CNN-Edge (modified Dense U-Net, growth rate 8, ELU, batch renorm) outputs an edge probability map; CNN-ROI selects the region where cells are reliably visible.
- Postprocessing: edge map smoothed with a scale set by the Fourier-estimated mean cell size, then watershed; border cells and cells with < 75 % area in the ROI discarded.
- ECD and CV (std/mean of cell area) need >= 6 cells; HEX uses inner cells only (>= 6), mimicking Topcon rules.
- Cross-entropy loss, Nadam, 150 epochs, flip augmentation.
- Authors note a direct parameter-regression network was less accurate in earlier experiments (see [[viguerasguillen2019regression]]).

## Results
- Percentage error ECD 2.5 %, CV 5.7 %, HEX 5.7 % vs Topcon 7.5 / 17.5 / 18.3 % (verified-from-source).
- MAE ECD 30.0 cells/mm2, CV 1.5, HEX 3.0 vs Topcon 118.7, 4.6, 10.3 (verified-from-source).
- No significant difference from manual (P > 0.05) (verified-from-source).
- Bland-Altman: CV inside 95 % limits for only 92.1 % (month 1) and 92.5 % (month 3) of images; HEX 91.2-94.9 % (verified-from-source).
- Error spread shrinks with detected cell count; CV overestimated below ~50 cells (verified-from-source).

## Limitations
- Author-stated: single center, FED only; guttae images need more labelled data; CV bias at low cell count.
- Reviewer: errors measured against manual masks on 240 x 528 crops; the paper shows the device (Topcon) itself disagrees with manual by 17-18 % on CV/HEX, so device labels are not the same quantity as manual morphometry.

## Relevance to CD / CV / HEX
Directly addresses all three metrics on the same device brand. Two lessons for CLEAR-EC: (1) device CV/HEX can deviate strongly from manual truth, consistent with our finding that device CV is about twice the Voronoi area CV from expert markers; (2) a few merged/missed cells distort HEX and CV most, consistent with our detector (miss 15 %, false 16.5 %) ruining area CV.

## What it suggests we try
Plot our CV/HEX residuals against the number of cells our detector finds inside the 538 x 408 label box; if residuals blow up at low counts (dark 2023-24 frames), down-weight the mask-derived features there and fall back to the image-regression head. Use Fourier mean-cell-size (spectral peak) as a cheap, segmentation-free CD/scale feature.

## Related notes
- [[viguerasguillen2019fcn]] — earlier CNN-Edge work on the same data
- [[viguerasguillen2019regression]] — direct CNN regression of ECD/CV/HEX
- [[viguerasguillen2022feedbackattn]] — follow-up for guttae images

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7445361
- https://www.github.com/jpviguerasguillen/corneal_endothelium_TVST (search snippet only)
