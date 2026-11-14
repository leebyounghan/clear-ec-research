---
id: marrugo2026review
type: paper
title: "Recent advances in corneal specular microscopy image analysis through artificial intelligence"
authors: [Andres G. Marrugo, Fernando Quintero, Alejandro Tello, Angelica M. Prada, Virgilio Galvis, Lenny A. Romero]
year: 2026
venue: "PLOS Digital Health"
doi: "10.1371/journal.pdig.0001305"
arxiv: ""
pmcid: "PMC13038023"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC13038023/"
citation: "Marrugo AG, Quintero F, Tello A, Prada AM, Galvis V, Romero LA. Recent advances in corneal specular microscopy image analysis through artificial intelligence. PLOS Digit Health. 2026;5(3):e0001305. doi:10.1371/journal.pdig.0001305"
tier: candidate
group: t1-endothelium-segmentation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: review
tldr: "Up-to-date (March 2026) review of classical and AI analysis of specular microscopy: U-Net variants dominate, GAN data generation and signed-distance regression are niche, no transformer/SAM work is covered, and small analysed areas plus device-software bias in Fuchs remain open problems."
relevance_clearec: medium
relevance_metrics: [CD, CV, HEX]
relevance_note: "Map of the field up to 2026; confirms no published method regresses device CV/HEX at scale and that results from small analysed areas are unstable (multi-acquisition median proposed)."
code_release: none
weights_release: none
data_scale: "review; Table 1 covers 9-10 studies with 90 to 4172 images"
key_numbers:
  - "Table 1: Daniel 2019 R2 0.96, PPV 84 %; Kucharski 2021 Dice 0.868, R2 0.929; Vigueras-Guillen 2022 MAE ECD 23.16; Tey 2025 Dice 0.86 cells / 0.90 guttae, ECD r 0.91 (verified-from-source)"
  - "Fuchs: device software mean ECD 2216 vs AI estimated ECD 1322 cells/mm2 (Prada 2024), because guttae areas are excluded by the device (verified-from-source)"
  - "Four acquisitions of one region gave AI ECD 1100, 1426, 1337, 1296 cells/mm2; median proposed (verified-from-source)"
  - "No MAPE values and no CV/HEX device-vs-AI numbers in the review (verified-from-source)"
results: []
suggests_try: "Exploit the 'median over acquisitions' idea as test-time augmentation: predict on several crops/flips/brightness variants of the frame and take the median per metric, which should reduce variance on dark, vignetted 2023-24 frames."
tags: [review, specular-microscopy, fuchs, guttae, gan, u-net]
search_keywords: [review AI corneal specular microscopy 2026, endothelial cell density AI review]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Narrative review; no quantitative synthesis; omits transformer/foundation-model work and CV/HEX error analysis", locator: "Table 1"}
---

# Recent advances in corneal specular microscopy image analysis through AI

## TL;DR
A 2026 narrative review from the Cartagena (Colombia) group covering classical methods, U-Net-based segmentation, semi/self-supervised learning, GAN data generation, signed-distance-map regression and multi-acquisition estimation. It finds that datasets are small, device software overestimates ECD in Fuchs dystrophy (guttae excluded), and small analysed areas give unstable estimates.

## Citation and link
- Marrugo AG, et al. PLOS Digit Health. 2026;5(3):e0001305. doi:10.1371/journal.pdig.0001305
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13038023/

## Problem and motivation
Summarise how AI changed endothelial morphometry from specular microscopy and what blocks clinical use.

## Data
Review. Table 1 lists Daniel (385 images), Shilpashree (246), Kucharski (119), Vigueras-Guillen (1203), Sierra (90), Foo (775), Sanchez (90), Tey (2193), Qu (4172).

## Method
Narrative review with a comparison table; no meta-analysis.

## Results
- Table 1 metrics as in key_numbers (verified-from-source).
- Device software overestimates ECD in FECD (2216 vs 1322 cells/mm2) (verified-from-source).
- Multi-acquisition example: 1100-1426 cells/mm2 across four acquisitions of the same region (verified-from-source).
- No transformer, SAM or foundation-model study is described; Qu 2024 "ECCT" is listed without detail (verified-from-source).

## Limitations
- Author-stated: data limitations, site shortcuts (illumination, calibration), image quality, translation barriers.
- Reviewer: no CV/HEX quantitative synthesis; nothing on device-label regression like CLEAR-EC.

## Relevance to CD / CV / HEX
Confirms the CLEAR-EC setting (device labels, ~150 cells in a chosen region) is under-studied. The site-shortcut warning (illumination, calibration) matches our two-source split (CD error 7.11 old vs 5.50 new; darker 2023-24 images).

## What it suggests we try
Median over test-time variants (crops covering the likely label box, flips, gamma) per metric; and check that per-source calibration does not let the model shortcut on illumination.

## Related notes
- [[daniel2019unet]] — Table 1 entry
- [[viguerasguillen2022feedbackattn]] — Table 1 entry
- [[sierra2023sdm]] — signed distance map regression
- [[kucharski2023gan]] — GAN-generated training data
- [[tey2025widefield]] — widefield U-Net, ECD r 0.91

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13038023/
