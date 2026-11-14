---
id: tey2025widefield
type: paper
title: "Deep Learning Analysis of Widefield Cornea Endothelial Imaging in Fuchs Dystrophy"
authors: [Kai Yuan Tey, Brian Juin Hsein Lee, Clarissa Ng, Qiu Ying Wong, Satish K. Panda, Amrit Dash, et al.]
year: 2025
venue: "Ophthalmology Science"
doi: "10.1016/j.xops.2025.100914"
arxiv: ""
pmcid: "PMC12548096"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC12548096"
citation: "Tey KY, Hsein Lee BJ, Ng C, Wong QY, Panda SK, Dash A, et al. Deep Learning Analysis of Widefield Cornea Endothelial Imaging in Fuchs Dystrophy. Ophthalmol Sci. 2025;6(1):100914. doi:10.1016/j.xops.2025.100914"
tier: candidate
group: t1-endothelium-segmentation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "A small two-stage U-Net segments guttae and cells on Nidek CEM-530 widefield images (Dice 0.86 cells, 0.90 guttae), but its ECD, CV and HEX differ grossly from the device's in-built values (CV 48.9 vs 30.2, HEX 46.2 vs 63.8, ECD 3134 vs 1977)."
relevance_clearec: medium
relevance_metrics: [CD, CV, HEX]
relevance_note: "Cautionary: a decent-Dice segmentation with its own CD formula and polygon counting lands far from device values on all three metrics; device labels must be matched by definition, not by segmentation quality."
code_release: unknown
weights_release: unknown
data_scale: "100 training images (50 FECD + 50 control eyes); test 354 images from 55 FECD eyes; 1839 widefield images from 155 eyes overall"
key_numbers:
  - "Dice 0.86 +/- 0.04 (cells), 0.90 +/- 0.04 (guttae) (verified-from-source)"
  - "DLN vs in-built device analysis, 354 images: CV 48.90 vs 30.16, HEX 46.16 vs 63.76, ECD 3133.52 vs 1976.67 cells/mm2, all P < 0.001 (verified-from-source)"
  - "Central ECD 2633.12 (DLN) vs 1728.58 (manual in-built) (from-abstract)"
  - "DLN ECD = detected cells / detectable area, not 1e6 / mean cell area (verified-from-source)"
  - "419 of 1485 images (28.2 %) not assessable (verified-from-source)"
results:
  - {task: "ECD/CV/HEX vs device in-built analysis", dataset: "354 CEM-530 images, 55 FECD eyes", metric: "mean value", value: "ECD 3133.52 / CV 48.90 / HEX 46.16", comparator: "in-built manual analysis", comparator_value: "1976.67 / 30.16 / 63.76", locator: "Results table", provenance: verified-from-source}
suggests_try: "Keep our CD feature on the label's own formula (1e6 / mean cell area of the analysed cells, integer area) and compute HEX as six-neighbour share and CV on the same cell set the device would use; never feed raw polygon-vertex HEX or count/area density without calibrating to labels."
tags: [widefield, nidek, fuchs, guttae, u-net, definition-mismatch]
search_keywords: [widefield specular microscopy deep learning, CEM-530 Fuchs deep learning, guttae segmentation U-Net]
uses_dataset: [Singapore National Eye Centre widefield FECD set]
uses_method: [U-Net, Canny, Ramer-Douglas-Peucker polygon classification]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Model may overestimate ECD; needs validation in advanced FECD; small training set", locator: "Discussion"}
  - {stated_by: reviewer, note: "Inconsistent training-set and ECD-difference figures; Pearson r 0.91 quoted by a review is not in the article text", locator: "Results/Discussion"}
---

# Deep Learning Analysis of Widefield Cornea Endothelial Imaging in Fuchs Dystrophy

## TL;DR
Tey et al. segment guttae and cells on Nidek widefield specular images with a lightweight two-stage U-Net and compute ECD, CV and HEX. Segmentation Dice is fine (0.86), yet all three metrics disagree strongly with the device's in-built analysis: CV is 19 points higher, HEX 18 points lower and ECD about 1150 cells/mm2 higher.

## Citation and link
- Tey KY, et al. Ophthalmol Sci. 2025;6(1):100914. doi:10.1016/j.xops.2025.100914
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12548096

## Problem and motivation
Widefield specular microscopy produces 15 images per eye; manual analysis takes 30-40 min per image in Fuchs dystrophy.

## Data
Nidek CEM-530; 15 images per eye (central, paracentral, peripheral); resized to 400 x 600. Training: 100 images (50 FECD, 50 control eyes); test: 354 images, 55 eyes.

## Method
Two U-Nets (~0.32 M params): (1) guttae vs cell region, (2) cells in the masked image; Canny edges, area threshold, Ramer-Douglas-Peucker polygon classification for sides. ECD = cell count / detectable area.

## Results
- Dice 0.86 (cells), 0.90 (guttae) (verified-from-source).
- CV 48.90 vs 30.16; HEX 46.16 vs 63.76; ECD 3133.52 vs 1976.67 (verified-from-source).
- Correlations and Bland-Altman limits: not reported numerically.

## Limitations
- Author-stated: ECD may be overestimated; small cohort; only eyes without definite edema.
- Reviewer: comparator is the device's own analysis, which itself is not ground truth; polygon-vertex HEX from Canny contours inflates side counts.

## Relevance to CD / CV / HEX
Mirror image of our problem: the label is a device value with its own definitions. Polygon-based HEX and count/area ECD from DL masks are biased against device values; consistent with our observation that device CV is about twice Voronoi CV (definitions differ in both directions depending on the pipeline).

## What it suggests we try
Audit every hand-crafted feature for definition match against the label formula (CD = 1e6/integer mean area; HEX = six-neighbour share); when definitions cannot be matched, feed the feature to the regressor rather than using it as a direct estimate.

## Related notes
- [[marrugo2026review]] — lists this study in Table 1
- [[sierra2023sdm]] — device software vs manual on Topcon
- [[viguerasguillen2022feedbackattn]] — HEX vertex vs neighbour method

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12548096
- https://pubmed.ncbi.nlm.nih.gov/41140904/ (via PubMed metadata tool)
