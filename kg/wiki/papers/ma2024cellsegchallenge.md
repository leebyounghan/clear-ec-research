---
id: ma2024cellsegchallenge
type: paper
title: "The multimodality cell segmentation challenge: toward universal solutions"
authors: [Jun Ma, Ronald Xie, Shamini Ayyadhury, Cheng Ge, Anubha Gupta, Ritu Gupta, et al.]
year: 2024
venue: "Nature Methods"
doi: "10.1038/s41592-024-02233-6"
arxiv: "2308.05864"
pmcid: "PMC11210294"
url: "https://arxiv.org/html/2308.05864v2"
citation: "Ma J, Xie R, Ayyadhury S, Ge C, Gupta A, Gupta R, et al. The multimodality cell segmentation challenge: toward universal solutions. Nat Methods. 2024;21(6):1103-1113. doi:10.1038/s41592-024-02233-6"
tier: candidate
group: t4-cell-foundation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: peer-reviewed
tldr: "NeurIPS 2022 challenge with 1,000 labelled and 1,725 unlabelled microscopy images across four modalities; the top three teams all tried the unlabelled images (consistency, pseudo-labels, low-uncertainty Listen2Student) but none showed a notable gain; winner median F1 89.7 % on a 422-image holdout."
relevance_clearec: medium
relevance_metrics: [CV, HEX]
relevance_note: "Field-level evidence that unlabelled images rarely help a segmenter when labelled in-domain data exist; what helped were external pretraining with replay, diverse intensity augmentation, multi-head distance/flow outputs and TTA."
code_release: open
weights_release: open
data_scale: "1,000 labelled train, 1,725 unlabelled, 101 tuning, 422 holdout test, 157 post-challenge yeast/bacteria images"
key_numbers:
  - "Holdout median F1: T1-osilab 89.7 %, T2-sribdmed 84.5 %, T3-cells 84.4 % (verified-from-source)"
  - "Unlabelled data: 'none of the employed methods demonstrated a notable enhancement'; no numbers given (verified-from-source)"
  - "T1 exceeded Cellpose-pretrain by 24.4 % and Omnipose-pretrain by 58.9 % (relative, median F1) (verified-from-source)"
  - "Post-challenge set (N=157) medians: T1 95.0 %, T2 88.7 %, T3 93.3 % (verified-from-source)"
results:
  - {task: "cell instance segmentation", dataset: "holdout test (422)", metric: "median F1", value: "89.7 %", comparator: "T2-sribdmed", comparator_value: "84.5 %", locator: "Results", provenance: "verified-from-source"}
suggests_try: "Spend effort on intensity/vignetting augmentation, flow/distance multi-head outputs and TTA rather than on generic semi-supervised losses; if pseudo-labels are used, filter them with an outside signal (our image-level CD/HEX)."
tags: [challenge, benchmark, cell-segmentation, semi-supervised, unlabelled-data]
search_keywords: [NeurIPS 2022 CellSeg challenge, multimodality cell segmentation, weakly supervised cell segmentation]
uses_dataset: [NeurIPS 2022 CellSeg]
uses_method: [consistency regularisation, pseudo-labelling, Listen2Student, TTA]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Effective use of unlabelled data remains an open question", locator: "Discussion"}
  - {stated_by: reviewer, note: "Reported T1 IQR differs between two places in the v2 text (84.1-94.8 % vs 36.7-82.4 %); v2 preprint read, not the journal version", locator: "Results"}
---

# The multimodality cell segmentation challenge: toward universal solutions

## TL;DR
The NeurIPS 2022 cell segmentation challenge gave 1,000 labelled and 1,725 unlabelled images across brightfield, fluorescence, phase contrast and DIC. The winner (MEDIAR, T1-osilab) reached 89.7 % median F1 on 422 holdout images, well above Cellpose and Omnipose. All top three teams tried to exploit unlabelled images, yet none showed a notable improvement from doing so.

## Citation and link
- Ma J, Xie R, Ayyadhury S, et al. Nat Methods. 2024;21(6):1103-1113. doi:10.1038/s41592-024-02233-6 (PMC11210294)
- https://arxiv.org/html/2308.05864v2

## Problem and motivation
Benchmark universal, parameter-free cell segmentation across modalities with limited labels.

## Data
Train 1,000 labelled (300 BF, 300 fluorescence, 200 PC, 200 DIC) + 1,725 unlabelled; tuning 101; holdout 422 incl. two WSIs; post-challenge 157 yeast/bacteria images.

## Method
Overview of submissions. T1: SegFormer + MA-Net, described here as using consistency between clean and degraded unlabelled images plus a reconstruction head; T2: modality classifier + group-specific ConvNeXt models with pseudo-label learning; T3: contour proposal network with Listen2Student low-uncertainty pseudo-labels.

## Results
- Median F1: 89.7 % / 84.5 % / 84.4 % (verified-from-source).
- No notable gain from unlabelled data for any method; not quantified (verified-from-source).
- Common success factors: intensity normalisation, patch sampling, diverse augmentation (cell-wise intensity perturbation, Mosaic), external pretraining with replay, multi-head distance/gradient outputs, TTA (T1) (verified-from-source, qualitative).

## Limitations
- Author-stated: unlabelled-data use unresolved.
- Reviewer: the MEDIAR paper itself says the final model used only labelled images and that its semi-supervised variants lowered F1 (see [[lee2022mediar]]).

## Relevance to CD / CV / HEX
Tempers expectations for self-training on our 8953 frames. Our setting differs (no in-domain masks; supervision available only at image level), so the transferable lesson is: generic unlabelled losses are weak, external checks on pseudo-labels are needed.

## What it suggests we try
Prioritise brightness/vignetting augmentation and TTA for any segmenter we train; filter pseudo-masks by image-label agreement rather than using plain consistency loss.

## Related notes
- [[lee2022mediar]] — winner; its own ablation shows unlabelled data hurt
- [[reith2025selfadapt]] — source-free adaptation that does help a generalist model
- [[pachitariu2025cellposesam]] — trained partly on this challenge's data

## Sources
- https://arxiv.org/html/2308.05864v2
- PubMed metadata PMID 38532015
