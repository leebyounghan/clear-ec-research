---
id: huang2017autovsmanual
type: paper
title: "Comparison of manual & automated analysis methods for corneal endothelial cell density measurements by specular microscopy"
authors: [Jianyan Huang, Jyotsna Maram, Tudor C Tepelus, Cristina Modak, Ken Marion, SriniVas R Sadda, et al.]
year: 2018
venue: "Journal of Optometry"
doi: "10.1016/j.optom.2017.06.001"
arxiv: ""
pmcid: "PMC6039579"
url: "https://pubmed.ncbi.nlm.nih.gov/28797649/"
citation: "Huang J, Maram J, Tepelus TC, Modak C, Marion K, Sadda SR, Chopra V, Lee OL. Comparison of manual & automated analysis methods for corneal endothelial cell density measurements by specular microscopy. J Optom. 2018;11(3):182-191. doi:10.1016/j.optom.2017.06.001"
tier: core
group: t2-device-morphometry
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "On Konan NSP-9900 images analysed with CellChek software, fully- and semi-automated ECD agreed with the manual Center Method in young normal eyes but overestimated ECD in older and glaucomatous eyes, i.e. when polymegethism or cell size is high."
relevance_clearec: medium
relevance_metrics: [CD, CV]
relevance_note: "Device bias against manual counts depends on CV and cell size, so CD label error is heteroscedastic and correlated with CV; relevant if CLEAR-EC labels mix automatic and manual CellChek modes."
code_release: none
weights_release: none
data_scale: "94 control + 106 glaucomatous eyes, Konan NSP-9900, Doheny reading-centre graders"
key_numbers:
  - "Center vs Flex-Center ECD: no difference in guttata (p = 0.48) or low ECD (p = 0.11) (from-abstract)"
  - "Fully-automated vs manual Center ECD in controls < 40 y: no difference (p = 0.32) (from-abstract)"
  - "Older controls and glaucoma: ECD overestimated by fully-automated (p = 0.034) and semi-automated (p = 0.025) methods vs manual (from-abstract)"
  - "Effect sizes (cells/mm2) not given in the abstract (not reported) (from-abstract)"
results: []
suggests_try: "Check on OOF whether CD residuals grow with predicted CV and with predicted mean cell area (old, polymegethous corneas); if so, a two-input CD correction (CD, CV) fitted by OOF MAPE is a cheap way to absorb this device behaviour."
tags: [konan, cellchek, automated-vs-manual, ecd-bias, polymegethism]
search_keywords: ["Konan CellChek fully automated overestimate ECD", "manual center method vs automated specular", "Doheny reading center specular"]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Fully-automated analysis alone is discouraged when accurate ECD is needed.", locator: "Abstract, conclusions"}
  - {stated_by: reviewer, note: "Abstract-level read; ECD only (no CV/HEX agreement); clinic eyes, not donor corneas.", locator: "Abstract"}
---

# Comparison of manual & automated analysis methods for corneal endothelial cell density measurements by specular microscopy

## TL;DR
Konan CellChek automatic methods match manual Center Method ECD in young normal eyes but overestimate ECD when cells are large or polymegethous (older and glaucomatous eyes).

## Citation and link
- Huang J et al. J Optom 2018;11(3):182-191. doi:10.1016/j.optom.2017.06.001
- https://pubmed.ncbi.nlm.nih.gov/28797649/ (abstract via PubMed)

## Problem and motivation
How reliable are automated ECD readouts versus validated manual methods, and what predicts disagreement?

## Data
Sharp central images of 94 control and 106 glaucomatous eyes, Konan NSP-9900; trained graders with Konan CellChek software (fully automated, semi-automated, Center Method; Flex-Center for low-count or guttata images).

## Method
ECD comparison, absolute error, effect of age, cell count, cell size and size variation.

## Results
See key_numbers (from-abstract).

## Limitations
- Author-stated: do not rely on fully automated analysis alone.
- Reviewer: no CV/HEX agreement; effect sizes not in abstract.

## Relevance to CD / CV / HEX
- CD: device-vs-manual bias depends on polymegethism (CV) and cell size, so a CD correction conditioned on CV is plausible device behaviour.
- CV: high-CV corneas are exactly where the automatic and manual modes diverge.

## What it suggests we try
Test a joint CD correction g(CD_pred, CV_pred) on OOF; keep it only if OOF CD MAPE drops.

## Related notes
- [[benetz2024donorecd]] — donor images; automatic-vs-manual bias grows with ECD.
- [[patel2010flexcenter]] — method effects on CV.

## Sources
- https://pubmed.ncbi.nlm.nih.gov/28797649/
- Companion (not a separate note): Huang J et al. Eye Contact Lens 2018;44 Suppl 1:S144-S150, doi:10.1097/ICL.0000000000000362 — machine-generated Konan NSP-9900 ECD was higher than manual in all cases; semiautomatic analysis was clinically acceptable.
