---
id: doughty2020cvcells
type: paper
title: "Importance of standardizing the number of cells measured for coefficient of variation (COV) estimates of corneal endothelial cell area values as relevant to contact lens wear"
authors: [Michael J Doughty]
year: 2020
venue: "Contact Lens & Anterior Eye"
doi: "10.1016/j.clae.2020.05.005"
arxiv: ""
pmcid: ""
url: "https://pubmed.ncbi.nlm.nih.gov/32513521/"
citation: "Doughty MJ. Importance of standardizing the number of cells measured for coefficient of variation (COV) estimates of corneal endothelial cell area values as relevant to contact lens wear. Cont Lens Anterior Eye. 2020;44(3):101336. doi:10.1016/j.clae.2020.05.005"
tier: candidate
group: t2-device-morphometry
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "CV estimates from manually measured cell areas stabilise only near 90 cells and their uncertainty grows with polymegethism (about +-1 % at CV 26 % to +-5 % at CV 57 %, relative to the >100-cell value); the number of cells should be the same across a data set."
relevance_clearec: medium
relevance_metrics: [CV]
relevance_note: "Gives the shape of CV label noise: it scales with the CV itself and with fewer cells, so CV label noise is heteroscedastic (larger in the upper CV decile)."
code_release: none
weights_release: none
data_scale: "4 sets x 20 non-contact specular images, graded by polymegethism (grade 0-3)"
key_numbers:
  - "Group-mean COV 25.8, 33.1, 45.1, 56.8 % for grades 0-3 (from-abstract)"
  - "Reliability with 90 cells: about +-1.0, +-2.7, +-3.6, +-5.0 % (relative to the >100-cell value) (from-abstract)"
  - "Reliability with 75 cells: +-3.4, +-5.3, +-6.1, +-7.9 % (from-abstract)"
  - "Recommendation: 75-100 cells per image, same number across the data set (from-abstract)"
results: []
suggests_try: "Give the CV distribution head a label-noise width that grows with CV (for example sigma proportional to label, or larger above CV 0.45) instead of a constant, and decode the 1/y-weighted median; check on OOF whether the top CV decile error falls without hurting the middle."
tags: [cv, sampling-error, polymegethism, number-of-cells, label-noise]
search_keywords: ["coefficient of variation endothelium number of cells", "polymegethism estimate reliability", "Doughty COV cell count"]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Subsets are nested within the same >100-cell set, so the quoted reliability understates independent-sample error; manual area tracing, not a device algorithm; abstract-level read.", locator: "Abstract, methods"}
---

# Importance of standardizing the number of cells measured for COV estimates

## TL;DR
Adding cells progressively from 2 to 100 shows that CV estimates converge only near 90 cells, and that the residual uncertainty is larger in polymegethous endothelia. Data sets should use the same number of cells per image.

## Citation and link
- Doughty MJ. Cont Lens Anterior Eye 2020;44(3):101336. doi:10.1016/j.clae.2020.05.005
- https://pubmed.ncbi.nlm.nih.gov/32513521/ (abstract via PubMed)

## Problem and motivation
CV is reported from widely varying numbers of cells; how much does that matter?

## Data
Four sets of 20 non-contact specular images from Caucasian individuals, grouped by polymegethism grade (none to substantial).

## Method
Cell areas measured manually; CV computed from the first 2..100 cells and compared with the value from > 100 cells.

## Results
See key_numbers (from-abstract).

## Limitations
- Reviewer: nested subsampling understates the independent sampling error; manual tracing rather than device software.

## Relevance to CD / CV / HEX
- CV: label noise scales with CV and depends on how many cells the technician dotted (our marker sets: median 148, p10-p90 119-158). The outer CV deciles hold a third of our CV error; part of that is this heteroscedastic noise and is not learnable.

## What it suggests we try
Heteroscedastic label-noise width in the CV head ([[gao2017dldl]], [[kendall2017uncertainties]]) and MAPE-optimal decoding ([[demyttenaere2016mape]]).

## Related notes
- [[doughty2018regional]] — regional variability of CV on Topcon images.
- [[mccarey2008review]] — cell-count recommendations.

## Sources
- https://pubmed.ncbi.nlm.nih.gov/32513521/
