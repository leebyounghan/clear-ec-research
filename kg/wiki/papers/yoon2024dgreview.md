---
id: yoon2024dgreview
type: paper
title: "Domain Generalization for Medical Image Analysis: A Review"
authors: [Jee Seok Yoon, Kwanseok Oh, Yooseung Shin, Maciej A. Mazurowski, Heung-Il Suk]
year: 2024
venue: "arXiv"
doi: ""
arxiv: "2310.08598"
pmcid: ""
url: "https://arxiv.org/html/2310.08598v3"
citation: "Yoon JS, Oh K, Shin Y, Mazurowski MA, Suk HI. Domain Generalization for Medical Image Analysis: A Review. arXiv:2310.08598 (v3, Dec 2024)"
tier: candidate
group: t7-domain-shift
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html]
evidence: review
tldr: "Workflow-oriented review of domain generalisation in medical imaging with a data-level / feature-level / model-level / analysis-level taxonomy; no quantitative benchmark, no regression tasks; notes histogram matching scored best among common DG methods in one cited benchmark and lists intensity normalisation, augmentation and ensembles as data/model-level tools."
relevance_clearec: low
relevance_metrics: [CD, CV, HEX]
relevance_note: "Map of options for our two-source acquisition shift; it confirms that data-level methods (normalisation, augmentation) are the efficient first step, but offers no regression-specific or quantitative guidance."
code_release: none
weights_release: none
data_scale: "Review; 1,621 papers in its database"
key_numbers:
  - "Taxonomy: data-level, feature-level, model-level, analysis-level (verified-from-source)"
  - "Cites a benchmark where histogram matching had the highest performance among some commonly used DG methods (verified-from-source)"
  - "No quantitative comparison table; regression tasks not discussed (verified-from-source)"
results: []
suggests_try: "As a cheap baseline for the darker 2023-24 frames, test histogram matching of each image to a reference intensity histogram from the other source (alongside flat-field correction and augmentation) and compare per-source MAPE."
tags: [review, domain-generalisation, medical-imaging, taxonomy]
search_keywords: [domain generalization medical image analysis review, DG taxonomy, histogram matching DG]
uses_dataset: [Camelyon17-WILDS, M&Ms, MIDOG, RETOUCH]
uses_method: [intensity normalisation, histogram matching, augmentation, feature alignment, ensembles]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Benchmarks for DG need expansion; source-limited and concept-shift settings remain open", locator: "Future directions"}
  - {stated_by: reviewer, note: "No numbers to choose between methods; TTA treated only as a related task", locator: "whole review"}
---

# Domain Generalization for Medical Image Analysis: A Review

## TL;DR
A broad review that organises domain generalisation methods along the medical imaging workflow (data, features, model, analysis) and discusses datasets and future directions. It is useful as a map but gives no quantitative comparison and does not cover regression.

## Citation and link
- Yoon JS, Oh K, Shin Y, Mazurowski MA, Suk HI. arXiv:2310.08598 (v3, 8 Dec 2024). Peer-reviewed venue not shown on the page read.
- https://arxiv.org/html/2310.08598v3

## Problem and motivation
Models trained on some sites/scanners must work on unseen ones without target data.

## Data
Review; discusses Camelyon17-WILDS, M&Ms (5 sites, 5 scanners), MIDOG (5 hospitals), RETOUCH (70 OCT volumes, 3 sites).

## Method
Taxonomy: data manipulation (intensity normalisation, histogram matching, frequency methods), augmentation (randomisation, adversarial, generative), feature alignment/disentanglement, model strategies (meta-learning, self-supervision, ensembles, distillation), analysis (interpretability, causality).

## Results
- No benchmark numbers; one cited benchmark found histogram matching best among some common DG methods (verified-from-source).
- Ensembles listed as increasing robustness at high compute cost (verified-from-source).

## Limitations
- Author-stated: benchmark gap.
- Reviewer: little actionable guidance for a two-source grayscale regression problem.

## Relevance to CD / CV / HEX
Frames our options for the old/new source shift (darker, vignetted 2023-24 frames; CD error 7.11 vs 5.50): input normalisation (flat-field, histogram matching), appearance augmentation, ensembles. Specific evidence comes from [[zhang2020bigaug]], [[tellez2019augnorm]] and [[peng2017basic]].

## What it suggests we try
Add histogram matching to a common reference as a third cheap preprocessing arm next to flat-field correction and augmentation, judged by per-source OOF MAPE.

## Related notes
- [[zhang2020bigaug]] — data-level DG with numbers
- [[tellez2019augnorm]] — augmentation vs normalisation, controlled study
- [[karani2021tta]] — test-time adaptation reference cited in the review

## Sources
- https://arxiv.org/html/2310.08598v3
