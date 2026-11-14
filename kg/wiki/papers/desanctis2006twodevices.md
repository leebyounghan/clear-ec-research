---
id: desanctis2006twodevices
type: paper
title: "Corneal endothelium evaluation with 2 noncontact specular microscopes and their semiautomated methods of analysis"
authors: [Ugo de Sanctis, Federica Machetta, Luca Razzano, Paola Dalmasso, Federico M Grignolo]
year: 2006
venue: "Cornea"
doi: "10.1097/01.ico.0000214217.83119.af"
arxiv: ""
pmcid: ""
url: "https://pubmed.ncbi.nlm.nih.gov/16783136/"
citation: "de Sanctis U, Machetta F, Razzano L, Dalmasso P, Grignolo FM. Corneal endothelium evaluation with 2 noncontact specular microscopes and their semiautomated methods of analysis. Cornea. 2006;25(5):501-506. doi:10.1097/01.ico.0000214217.83119.af"
tier: core
group: t2-device-morphometry
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "In 49 normal subjects, two examiners using the Topcon SP-2000P retraced method and the Konan CC7000 center method agreed well on cell size and density (ICC > 0.85) but poorly on CV and hexagonality (ICC < 0.65), and CV/HEX correlated at r < 0.20 between the two instruments."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Quantifies how much of CV/HEX is reader and device behaviour rather than biology; supports treating CV/HEX labels as heavily noisy."
code_release: none
weights_release: none
data_scale: "49 normal subjects, 2 examiners, 2 instruments"
key_numbers:
  - "Inter-examiner ICC > 0.85 for average cell size and ECD (from-abstract)"
  - "Inter-examiner ICC < 0.65 for CV and hexagonality (from-abstract)"
  - "Topcon vs Konan ECD mean difference -184 (examiner 1) and -228 cells/mm2 (examiner 2) (from-abstract)"
  - "CV and HEX between instruments: no mean difference (P > 0.05) but r < 0.20 (from-abstract)"
results: []
suggests_try: "Use these reliabilities to set realistic targets: if reader ICC for CV/HEX is below 0.65 in normal eyes, a model at CV r 0.69 is near the reader ceiling, so spend remaining time on loss/decoding (MAPE-optimal shrinkage, noise-aware heads) rather than on better cell finding."
tags: [konan, topcon, inter-observer, reproducibility, cv, hexagonality]
search_keywords: ["interexaminer reproducibility specular microscopy CV hexagonality", "Topcon SP-2000P Konan CC7000", "endothelial morphometry ICC"]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Normal eyes only (narrow CV/HEX range lowers ICC/r mechanically); abstract-level read; old devices.", locator: "Abstract"}
---

# Corneal endothelium evaluation with 2 noncontact specular microscopes and their semiautomated methods of analysis

## TL;DR
Two examiners, two devices (Topcon SP-2000P retraced method, Konan CC7000 center method), 49 normal subjects. Density and cell size are reproducible; CV and hexagonality are not (ICC < 0.65), and CV/HEX barely correlate between devices (r < 0.20).

## Citation and link
- de Sanctis U et al. Cornea 2006;25(5):501-6. doi:10.1097/01.ico.0000214217.83119.af
- https://pubmed.ncbi.nlm.nih.gov/16783136/ (abstract via PubMed)

## Problem and motivation
Are semiautomated analyses reproducible across examiners and interchangeable across instruments?

## Data
49 normal subjects; each eye imaged and analysed by 2 examiners on 2 instruments.

## Method
ICC, 95 % limits of agreement, Pearson and Spearman correlations.

## Results
See key_numbers (from-abstract). Topcon gave larger cell size and lower ECD (about -184 to -228 cells/mm2) than Konan for both examiners.

## Limitations
- Reviewer: restricted range in normal eyes depresses ICC and r; full text not read.

## Relevance to CD / CV / HEX
- CV/HEX: reader plus device behaviour explains a large share of label variance in normal eyes. This agrees with the coordinator's root-cause result (78-88 % of the CV/HEX residual is what expert centres in the true region miss).
- CD: systematic device offset of about 8-10 %.

## What it suggests we try
Treat CV/HEX as noisy targets: heteroscedastic or distributional heads and MAPE-optimal decoding ([[gao2017dldl]], [[demyttenaere2016mape]], [[kendall2017uncertainties]]) are the levers left, not segmentation.

## Related notes
- [[gasser2015konantopcon]] — same conclusion with newer Konan/Topcon devices.
- [[vidalvillegas2024eyebank]] — eye-bank Konan CellChek D: CV/HEX ICC <= 0.265.
- [[scarpa2016morphometry]] — inter-expert gap on identical cells.

## Sources
- https://pubmed.ncbi.nlm.nih.gov/16783136/
