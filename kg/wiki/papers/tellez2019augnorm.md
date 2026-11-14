---
id: tellez2019augnorm
type: paper
title: "Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology"
authors: [David Tellez, Geert Litjens, Peter Bandi, Wouter Bulten, John-Melle Bokhorst, Francesco Ciompi, et al.]
year: 2019
venue: "Medical Image Analysis"
doi: "10.1016/j.media.2019.101544"
arxiv: "1902.06543"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/1902.06543"
citation: "Tellez D, Litjens G, Bandi P, Bulten W, Bokhorst JM, Ciompi F, van der Laak J. Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. Med Image Anal. 2019;58:101544. doi:10.1016/j.media.2019.101544"
tier: core
group: t7-domain-shift
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html]
evidence: peer-reviewed
tldr: "Systematic single-centre-train / multi-centre-test study over 4 pathology tasks and 9 labs: colour augmentation (HED/HSV) is what drives external-centre performance; colour normalisation adds robustness across augmentation choices but the best setup used no normalisation (identity + HED-light, mean rank 1.2)."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "The best controlled evidence on 'normalise the input vs augment appearance' under acquisition shift; it says appearance augmentation is the must-have and normalisation (our flat-field correction) is optional, informing how to spend our remaining 2.5 days on the old/new source gap."
code_release: unknown
weights_release: unknown
data_scale: "Four 128x128-patch tasks (mitosis, lymph-node metastasis, prostate epithelium, 9-class colorectal tissue); train Radboudumc only; external test from 9 centres"
key_numbers:
  - "Best configuration identity normalisation + HED-light augmentation, mean rank 1.2 +- 0.4 (verified-from-source)"
  - "External AUCs of best setup: lymph-cwh 0.952, lymph-rh 0.946, mitosis-tupac 0.996, prostate-cedar 0.879, crc-heidelberg 0.895 (verified-from-source)"
  - "Identity + HSV-strong rank 4.7 vs network normalisation + HSV-strong rank 6.6 (verified-from-source)"
  - "Guideline: always use colour augmentation; light augmentation if normalising, strong if not (verified-from-source)"
results:
  - {task: "patch classification, external centres", dataset: "9 external centres, 4 tasks", metric: "mean rank", value: "1.2 (identity + HED-light)", comparator: "network normalisation + HSV-strong", comparator_value: "6.6", locator: "Results ranking table", provenance: "verified-from-source"}
suggests_try: "Prioritise strong brightness/gamma/vignetting augmentation for the regression backbones over input normalisation; if per-source flat-field correction is added, pair it with lighter augmentation, and pick by per-source OOF MAPE."
tags: [data-augmentation, normalisation, domain-shift, multi-centre, computational-pathology]
search_keywords: [stain augmentation vs normalization, HED augmentation, multi-centre generalisation pathology]
uses_dataset: [Camelyon-type lymph node, TUPAC mitosis, prostate, colorectal Kather]
uses_method: [HED augmentation, HSV augmentation, stain normalisation, network-based normalisation]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Normalisation made results more stable across augmentation types but was not needed for the top result", locator: "Discussion"}
  - {stated_by: reviewer, note: "Classification of H&E colour shifts; our shift is grayscale brightness/vignetting on a regression target, so transfer is by analogy", locator: "n/a"}
---

# Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology

## TL;DR
The authors trained CNNs on one centre and tested on nine external centres across four pathology tasks, crossing many augmentation and normalisation schemes. Colour augmentation (HED or HSV) was the decisive factor; normalisation did not add on top of good augmentation, and the overall best configuration used no normalisation with light HED augmentation.

## Citation and link
- Tellez D, Litjens G, Bandi P, et al. Med Image Anal. 2019;58:101544. doi:10.1016/j.media.2019.101544 (arXiv 1902.06543)
- https://ar5iv.labs.arxiv.org/html/1902.06543

## Problem and motivation
Lab-to-lab stain/scanner variation degrades CNNs; both augmentation and normalisation are used in practice without controlled comparison.

## Data
Mitosis (breast), lymph-node metastasis, prostate epithelium, colorectal 9-class tissue; train Radboudumc; test 9 external centres.

## Method
Augmentations: basic, morphology, brightness/contrast, HSV light/strong, HED light/strong. Normalisation: identity, grayscale, deconvolution, LUT, style transfer, proposed network-based. Ranked by external AUC.

## Results
- Best: identity + HED-light, mean rank 1.2 (verified-from-source).
- External AUCs 0.879-0.996 for the best setup (verified-from-source).
- Identity + HSV-strong 4.7 vs network + HSV-strong 6.6 (verified-from-source).

## Limitations
- Author-stated: normalisation gives stability, not peak performance.
- Reviewer: colour vs grayscale; classification vs regression.

## Relevance to CD / CV / HEX
Our old and new sources differ in brightness and vignetting (CD error 7.11 vs 5.50) and the hidden-test HEX was 0.40 MAPE worse than OOF. This paper, together with [[zhang2020bigaug]] (brightness and contrast the most useful single transforms), says the main lever is appearance augmentation at training time; explicit correction such as [[peng2017basic]] is a secondary, stabilising step.

## What it suggests we try
Retrain the main regressor with strong brightness/gamma/synthetic-vignetting augmentation (no normalisation) and, separately, flat-field + light augmentation; keep whichever lowers the old-new MAPE gap on OOF.

## Related notes
- [[zhang2020bigaug]] — stacked augmentation for unseen domains in radiology
- [[peng2017basic]] — explicit shading correction (normalisation route)
- [[yoon2024dgreview]] — survey placing these under data-level DG

## Sources
- https://ar5iv.labs.arxiv.org/html/1902.06543
- DOI confirmed via search result for the journal record
