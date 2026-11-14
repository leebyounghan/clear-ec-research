---
id: patel2010flexcenter
type: paper
title: "Comparison of flex-center, center, and corner methods of corneal endothelial cell analysis"
authors: [Sanjay V Patel, Jay W McLaren, Lori A Bachman, William M Bourne]
year: 2010
venue: "Cornea"
doi: "10.1097/ICO.0b013e3181cc7a60"
arxiv: ""
pmcid: ""
url: "https://pubmed.ncbi.nlm.nih.gov/20520534/"
citation: "Patel SV, McLaren JW, Bachman LA, Bourne WM. Comparison of flex-center, center, and corner methods of corneal endothelial cell analysis. Cornea. 2010;29(9):1042-1047. doi:10.1097/ICO.0b013e3181cc7a60"
tier: core
group: t2-device-morphometry
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "On identical cells, Konan's flex-center method (which keeps the outermost cells that the center method drops) reads CV about 3 points higher than center and corner methods in normal corneas, while ECD agrees within about 1-2 %; in abnormal (post-PK) corneas CV and HEX disagree across methods."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Shows that the device's analysis mode alone (which cells enter the polygon statistics) shifts CV by about 10 % relative on the same cells, an irreducible device-behaviour term in our labels."
code_release: none
weights_release: none
data_scale: "10 normal corneas + 10 post-penetrating-keratoplasty corneas, identical cells analysed by 3 methods"
key_numbers:
  - "Normal ECD: flex-center 2846 +- 248, center 2870 +- 253, corner 2892 +- 254 cells/mm2 (from-abstract)"
  - "Normal CV: flex-center 33 % +- 3, center 30 % +- 3, corner 30 % +- 3 (P < 0.001) (from-abstract)"
  - "Normal HEX: flex-center 60 % +- 6, center 60 % +- 8, corner 58 % +- 7 (P = 0.004) (from-abstract)"
  - "PK CV: flex-center 35 % +- 13, center 30 % +- 10, corner 35 % +- 15 (P = 0.02) (from-abstract)"
  - "PK HEX: flex-center 56 % +- 19, center 54 % +- 17, corner 43 % +- 23 (P = 0.02) (from-abstract)"
results: []
suggests_try: "When emulating the device from points, compute CV twice, interior cells only (center) and interior plus outermost cells (flex-center); see which tracks the label better on the 25 expert-marker frames, and use the winner (or both) as regressor features."
tags: [konan, flex-center, center-method, corner-method, method-agreement, cv, hexagonality]
search_keywords: ["flex-center method Konan", "center method outermost cells excluded", "corner method endothelial analysis agreement"]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Morphometric data are not accurate when cell morphology is very abnormal (post-PK).", locator: "Abstract, conclusions"}
  - {stated_by: reviewer, note: "Only 20 corneas; abstract-level read; no inter-observer component, so the method effect is isolated but not the reader effect.", locator: "Abstract"}
---

# Comparison of flex-center, center, and corner methods of corneal endothelial cell analysis

## TL;DR
Konan's center method drops the outermost dotted cells of a contiguous group; the flex-center variant keeps them. On the same cells, ECD agrees across methods, but CV is about 3 points higher with flex-center in normal corneas, and CV and HEX disagree across methods in abnormal corneas.

## Citation and link
- Patel SV et al. Cornea 2010;29(9):1042-7. doi:10.1097/ICO.0b013e3181cc7a60
- https://pubmed.ncbi.nlm.nih.gov/20520534/ (abstract via PubMed)

## Problem and motivation
Flex-center analyses more cells in sparse images but its agreement with the standard methods was unknown.

## Data
Endothelial images of 10 normal corneas and 10 corneas after penetrating keratoplasty; the identical cells analysed with flex-center, center and corner (standard) methods. Device: Konan software (flex-center is a Konan method).

## Method
Agreement in ECD, CV and HEX tested with Student-Newman-Keuls.

## Results
See key_numbers (all from-abstract). ECD differences in normal corneas are statistically significant but small (about 1.6 % between flex-center and corner). CV: flex-center 33 % vs 30 % for the other two in normal eyes.

## Limitations
- Author-stated: morphometric agreement is poor in abnormal corneas.
- Reviewer: small sample; full text not read.

## Relevance to CD / CV / HEX
- CV: including the outer ring of cells raises CV by about 10 % relative on identical cells, the same order as our whole CV MAPE (10.11). Whether the eye-bank technician used center or flex-center is a hidden per-image (or per-bank) factor in the label.
- HEX: method effect small in normal corneas (58-60 %), large in abnormal ones.
- CD: method effect about 1-2 %.

## What it suggests we try
Emulate both modes from points (interior only vs interior plus outer ring, see [[takahashi1996konanpatent]]) and feed both CV values to the regressor; a learned mix can absorb a per-source preference for one mode without keying on source IDs.

## Related notes
- [[takahashi1996konanpatent]] — why the center method drops peripheral cells (angle-gap rule).
- [[vidalvillegas2024eyebank]] — center vs flex-center in an eye bank, CV/HEX ICC <= 0.265.
- [[huang2017autovsmanual]] — center vs flex-center agree for ECD in low-count images.

## Sources
- https://pubmed.ncbi.nlm.nih.gov/20520534/
