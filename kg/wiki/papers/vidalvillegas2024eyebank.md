---
id: vidalvillegas2024eyebank
type: paper
title: "Corneal endothelial validation in the eye bank: differences in automated methods and repeatability"
authors: [B Vidal-Villegas, B Burgos-Blasco, P Fernandez-Vega, P Arriola-Villalobos, J A Gegundez-Fernandez, L Borrego-Sanz, et al.]
year: 2024
venue: "Journal Français d'Ophtalmologie"
doi: "10.1016/j.jfo.2023.09.017"
arxiv: ""
pmcid: ""
url: "https://pubmed.ncbi.nlm.nih.gov/37951743/"
citation: "Vidal-Villegas B, Burgos-Blasco B, Fernandez-Vega P, et al. Corneal endothelial validation in the eye bank: differences in automated methods and repeatability. J Fr Ophtalmol. 2024;47(3):104022. doi:10.1016/j.jfo.2023.09.017"
tier: core
group: t2-device-morphometry
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "On 54 donor corneas imaged with a Konan CellChek D in an eye bank, two graders reproduced ECD well (ICC 0.82 center, 0.78 flex-center) but CV and hexagonality poorly (ICC <= 0.265); flex-center read significantly higher ECD than center."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Closest published analogue of the CLEAR-EC label process (eye-bank donor corneas, Konan CellChek D, technician-chosen cells); shows CV/HEX labels are mostly grader noise."
code_release: none
weights_release: none
data_scale: "54 donor corneas, 2 independent graders, 2 Konan methods"
key_numbers:
  - "Mean ECD 2473.81 +- 378.22 cells/mm2 (from-abstract)"
  - "Inter-grader ICC for ECD: center 0.821, flex-center 0.784 (from-abstract)"
  - "Inter-grader ICC for CV and hexagonality <= 0.265 (from-abstract)"
  - "Center vs flex-center ECD correlation 0.678 and 0.745 (per grader); flex-center higher (P = 0.013), also in ECD < 2200 and 2200-2700 strata (P < 0.022) (from-abstract)"
  - "Center method: at least 30 contiguous cells marked; flex-center: an area is delineated and only cells inside it are counted (from-abstract)"
results: []
suggests_try: "Treat CV/HEX labels as mostly grader noise around a device-scale signal: compute the MAPE floor this implies (if ICC about 0.27, most label variance is noise) and prioritise MAPE-optimal shrinkage/decoding over trying to fit the outer CV deciles."
tags: [eye-bank, konan, cellchek-d, center-method, flex-center, inter-grader, cv, hexagonality]
search_keywords: ["eye bank specular microscopy repeatability", "Konan Cell Check D donor cornea", "flex-center center method donor"]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Decisions should not rest on small ECD differences.", locator: "Abstract, conclusions"}
  - {stated_by: reviewer, note: "Abstract-level read; exact CV/HEX ICC per method not given in the abstract; single Spanish eye bank; graders were ophthalmologists, not US eye-bank technicians.", locator: "Abstract"}
---

# Corneal endothelial validation in the eye bank: differences in automated methods and repeatability

## TL;DR
Two ophthalmologists analysed 54 donor corneas with the Konan CellChek D using both the center and flex-center methods. ECD was reproducible; CV and hexagonality were not (ICC <= 0.265). Flex-center gave higher ECD, especially at low ECD.

## Citation and link
- Vidal-Villegas B et al. J Fr Ophtalmol 2024;47(3):104022. doi:10.1016/j.jfo.2023.09.017
- https://pubmed.ncbi.nlm.nih.gov/37951743/ (abstract via PubMed)

## Problem and motivation
Eye banks rely on CellChek analyses for donor suitability; reproducibility and method agreement in donor tissue were unclear.

## Data
54 donor corneas, Konan CellChek D (Konan Medical USA), Hospital Clínico San Carlos tissue bank, Madrid.

## Method
Two independent graders; center method (>= 30 contiguous cells marked) and flex-center method (delineated area, only cells inside counted). ICC and correlation.

## Results
See key_numbers (from-abstract).

## Limitations
- Author-stated: small ECD differences should not drive decisions.
- Reviewer: abstract only; one bank; small n.

## Relevance to CD / CV / HEX
- CV/HEX: an inter-grader ICC near 0.27 means most of the label variance across donor corneas is grader-dependent (which cells, which region, which method). This is the setting of CLEAR-EC labels (US eye banks, Konan named as the example device on the challenge page). It supports the coordinator's finding that the remaining CV/HEX error is label noise plus device behaviour.
- CD: ICC about 0.8; method choice shifts ECD; consistent with our CD MAPE being the lowest of the three.

## What it suggests we try
Do not chase the outer CV/HEX deciles with reweighting; fit per-metric MAPE-optimal shrinkage (1/y-weighted median of a predicted distribution) and judge by OOF MAPE ([[demyttenaere2016mape]], [[delange2022mindgap]]).

## Related notes
- [[patel2010flexcenter]] — flex-center vs center on identical cells.
- [[desanctis2006twodevices]] — ICC < 0.65 for CV/HEX in normal clinic eyes.
- [[benetz2024donorecd]] — eye-bank donor images, Konan CellChek D, automatic ECD.

## Sources
- https://pubmed.ncbi.nlm.nih.gov/37951743/
