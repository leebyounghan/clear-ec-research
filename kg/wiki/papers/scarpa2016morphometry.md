---
id: scarpa2016morphometry
type: paper
title: "Development of a Reliable Automated Algorithm for the Morphometric Analysis of Human Corneal Endothelium"
authors: [Fabio Scarpa, Alfredo Ruggeri]
year: 2016
venue: "Cornea"
doi: "10.1097/ICO.0000000000000908"
arxiv: ""
pmcid: ""
url: "https://pubmed.ncbi.nlm.nih.gov/27310881/"
citation: "Scarpa F, Ruggeri A. Development of a Reliable Automated Algorithm for the Morphometric Analysis of Human Corneal Endothelium. Cornea. 2016;35(9):1222-1228. doi:10.1097/ICO.0000000000000908"
tier: core
group: t1-endothelium-segmentation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "A genetic-algorithm contour segmentation that uses the regular hexagonal shape prior gives ECD/HEX/CV within 0.6/3.1/5.3 % of manual on 30 Topcon SP-3000P images, while two experts differ from each other by 0.5/2.6/2.9 %."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Gives an inter-expert floor for the three metrics (ECD 0.5 %, HEX 2.6 %, CV 2.9 % mean absolute difference) and shows CV is the metric where automatic contours lag experts most."
code_release: unknown
weights_release: unknown
data_scale: "30 Topcon SP-3000P specular images, contours drawn by 2 experts"
key_numbers:
  - "Automatic vs manual mean percent absolute difference: ECD 0.6 %, pleomorphism (HEX) 3.1 %, polymegethism (CV) 5.3 % (from-abstract)"
  - "Expert vs expert: ECD 0.5 %, HEX 2.6 %, CV 2.9 % (from-abstract)"
  - "No significant automatic-vs-manual difference, P > 0.2, power >= 77 % (from-abstract)"
results:
  - {task: "ECD/HEX/CV from automatic contours", dataset: "30 Topcon SP-3000P images", metric: "mean percent absolute difference vs manual", value: "0.6 / 3.1 / 5.3", comparator: "second expert", comparator_value: "0.5 / 2.6 / 2.9", locator: "Abstract", provenance: from-abstract}
suggests_try: "Use the inter-expert gap (CV 2.9 %, HEX 2.6 % on the same cells) plus our ~150-cell sampling error (CV ~6.5 %, HEX ~7.5 %) as the noise floor when judging how much CV/HEX MAPE is still recoverable; spend the last days on CV (largest automatic-vs-expert gap here and largest gap to first place for us)."
tags: [genetic-algorithm, shape-prior, classical, topcon, inter-observer]
search_keywords: [automated morphometric analysis corneal endothelium, genetic algorithm endothelial contours, polymegethism pleomorphism automatic]
uses_dataset: []
uses_method: [genetic algorithm segmentation, hexagonal shape prior]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "30 clean images only; abstract-level reading; CV defined as fractional SD of cell area", locator: "Abstract"}
---

# Development of a Reliable Automated Algorithm for the Morphometric Analysis of Human Corneal Endothelium

## TL;DR
Scarpa and Ruggeri segment endothelial cell contours with a genetic algorithm that combines a regular-cell-shape prior with image intensity. On 30 Topcon SP-3000P images, automatic ECD, pleomorphism and polymegethism are within 0.6 %, 3.1 % and 5.3 % of manual values, close to the 0.5 %, 2.6 % and 2.9 % that separate two experts.

## Citation and link
- Scarpa F, Ruggeri A. Cornea. 2016;35(9):1222-1228. doi:10.1097/ICO.0000000000000908
- https://pubmed.ncbi.nlm.nih.gov/27310881/ (abstract via PubMed metadata)

## Problem and motivation
Reliable morphometry needs accurate contours of many cells; manual tracing is impractical in clinics.

## Data
30 images from a Topcon SP-3000P specular microscope; ground truth from contours drawn manually by 2 experts.

## Method
- Genetic algorithm that evolves a cell-contour tessellation, scored by agreement with pixel intensity and by the typical regularity (near-hexagonal shape) of endothelial cells.
- ECD = cells per area; pleomorphism = fraction of hexagonal cells; polymegethism = fractional SD of cell areas.

## Results
- Automatic vs manual: ECD 0.6 %, HEX 3.1 %, CV 5.3 % mean percent absolute difference (from-abstract).
- Expert vs expert: ECD 0.5 %, HEX 2.6 %, CV 2.9 % (from-abstract).
- P > 0.2 for all parameters, power >= 77 % (from-abstract).

## Limitations
- Author-stated: not available (abstract only).
- Reviewer: small, clean dataset; a hexagonal shape prior may bias HEX upward on irregular endothelium.

## Relevance to CD / CV / HEX
Shows the metric ordering of difficulty (ECD << HEX < CV) for automatic methods, matching our gap ordering (largest remaining gap on CV). The inter-expert floor (CV 2.9 %, HEX 2.6 %) is far below our OOF MAPE (CV 10.11, HEX 9.74), so most of our error comes from sampling (~150 cells, technician-chosen box) and device-definition mismatch rather than contour ambiguity.

## What it suggests we try
Report our CV/HEX error decomposed into (a) expert-marker-to-device mismatch on the 25 frames and (b) model-to-marker error; if (a) dominates, prioritise label-scale calibration (e.g., isotonic map from marker CV to device CV) over better segmentation.

## Related notes
- [[piorkowski2017segmentation]] — how the segmentation method moves CV/HEX
- [[viguerasguillen2019fcn]] — deep-learning counterpart on Topcon SP-1P

## Sources
- https://pubmed.ncbi.nlm.nih.gov/27310881/ (via PubMed metadata tool)
- https://research.unipd.it/handle/11577/3191107 (search snippet only)
