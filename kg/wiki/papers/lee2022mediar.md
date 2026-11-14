---
id: lee2022mediar
type: paper
title: "MEDIAR: Harmony of Data-Centric and Model-Centric for Multi-Modality Microscopy"
authors: [Gihun Lee, Sangmook Kim, Joonkee Kim, Se-Young Yun]
year: 2022
venue: "arXiv (NeurIPS 2022 Cell Segmentation Challenge)"
doi: ""
arxiv: "2212.03465"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/2212.03465"
citation: "Lee G, Kim S, Kim J, Yun SY. MEDIAR: Harmony of Data-Centric and Model-Centric for Multi-Modality Microscopy. arXiv:2212.03465. 2022 (NeurIPS 2022 Cell Segmentation Challenge)."
tier: core
group: t4-cell-foundation
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: preprint
tldr: "Winning NeurIPS 2022 cell segmentation pipeline: SegFormer + MA-Net with Cellpose-style flow head, external pretraining (7,242 labelled images), memory replay, modality-balanced sampling, flip TTA and two-model ensemble; validation F1 0.9067. Its own ablation shows consistency, reconstruction and pseudo-labelling on unlabelled images all LOWERED F1."
relevance_clearec: high
relevance_metrics: [CV, HEX]
relevance_note: "Direct evidence on our key open question: naive semi-supervised use of unlabelled images (pseudo-labels 0.8655, consistency 0.8720 vs baseline 0.8801) hurt when in-domain labels existed; supports using pseudo-labels only with an external quality filter (our image labels)."
code_release: open
weights_release: open
data_scale: "1,000 labelled challenge images (unlabelled set not used in final model) + 7,242 external labelled images (OmniPose 611, Cellpose 551, LiveCell 5,239, DSB 841)"
key_numbers:
  - "Final validation F1 0.9067 on 101 tuning images; no test numbers in the paper (verified-from-source)"
  - "Unlabelled-data ablation (Table 8): baseline 0.8801; consistency 0.8720; reconstruction 0.8798; pseudo-labelling 0.8655 (verified-from-source)"
  - "Cell memory replay 8:1 raises F1 0.8801 -> 0.9000 (phase 1); 7:2 gives 0.8881 (verified-from-source)"
  - "Inference: sliding window + stochastic TTA 0.9000 -> 0.9027 (phase 1), ensemble with flips and importance map 0.9067 (verified-from-source)"
  - "Encoder initialisation from pretrained weights: 0.6503 -> 0.8292 (verified-from-source)"
results:
  - {task: "cell instance segmentation", dataset: "NeurIPS 2022 CellSeg tuning set (101)", metric: "F1", value: "0.9067", comparator: "phase-1 baseline", comparator_value: "0.8801", locator: "Tables 4, 6", provenance: "verified-from-source"}
  - {task: "semi-supervised ablation", dataset: "NeurIPS 2022 CellSeg tuning set", metric: "F1", value: "0.8655 (pseudo-labelling)", comparator: "no unlabelled data", comparator_value: "0.8801", locator: "Table 8", provenance: "verified-from-source"}
suggests_try: "Do not run an unfiltered pseudo-label loop on the 8953 frames expecting gains; if we self-train a mask model, gate every pseudo-labelled frame by agreement with its image labels (CD within ~5 %, HEX within a few points) and keep a replay share of the 385 Topcon point images (MEDIAR's 8:1 replay) to avoid forgetting."
superseded: "2026-11-09: segmentation / self-training route closed for CV and HEX by the root-cause analysis; see body"
tags: [cell-segmentation, challenge-winner, semi-supervised-negative, memory-replay, TTA, cellpose-flows]
search_keywords: [MEDIAR, NeurIPS 2022 cell segmentation, osilab, cell memory replay, SegFormer MA-Net]
uses_dataset: [NeurIPS 2022 CellSeg, LiveCell, Cellpose, OmniPose, DSB 2018]
uses_method: [SegFormer, MA-Net, Cellpose gradient flows, memory replay, flip TTA, ensembling]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Unlabelled data not used; how to incorporate it is left as future work", locator: "Discussion"}
  - {stated_by: reviewer, note: "Challenge overview (Ma 2024) describes T1-osilab as using unlabelled data via consistency and reconstruction, which conflicts with this paper's statement that only the 1,000 labelled images are used", locator: "Ma 2024 Sec. methods vs MEDIAR Sec. 3"}
  - {stated_by: reviewer, note: "Validation-set ablations on 101 images; small differences (0.002-0.008 F1) may be noise", locator: "Tables 4-8"}
---

# MEDIAR: Harmony of Data-Centric and Model-Centric for Multi-Modality Microscopy

## TL;DR
MEDIAR won the NeurIPS 2022 multi-modality cell segmentation challenge with a SegFormer/MA-Net model predicting a cell mask and Cellpose-style gradient flows, pretrained on 7,242 public labelled images, fine-tuned with modality-balanced sampling and memory replay, and run with sliding-window flip TTA and a two-model ensemble. It reports validation F1 0.9067. Semi-supervised use of the 1,725 unlabelled images was tried and dropped because every variant lowered F1.

## Citation and link
- Lee G, Kim S, Kim J, Yun SY. arXiv:2212.03465 (2022).
- https://ar5iv.labs.arxiv.org/html/2212.03465 ; code https://github.com/Lee-Gihun/MEDIAR

## Problem and motivation
One model must segment cells across brightfield, fluorescence, phase contrast and DIC with heterogeneous modality frequencies and a strict time budget.

## Data
1,000 labelled challenge images; external pretraining: OmniPose 611, Cellpose 551, LiveCell 5,239 (1,686,352 cells), DSB 841 (37,333 cells), total 7,242. Validation = 101 tuning images.

## Method
- Model: SegFormer encoder + MA-Net decoder, two heads (BCE cell mask; MSE gradient flows, lambda 0.5), Cellpose flow tracking.
- Data-centric: k-means (40 clusters) on embeddings to discover modalities, over-sample minor ones; cell-aware augmentation; memory replay of public data in each batch (8:1).
- Inference: 512x512 windows with 0.6 overlap, flip TTA, Gaussian importance map, ensemble of phase-1 and phase-2 models.

## Results
- Final validation F1 0.9067 (verified-from-source).
- Unlabelled data (Table 8): baseline 0.8801; consistency 0.8720; reconstruction 0.8798; pseudo-labelling 0.8655 (verified-from-source).
- Replay 9:0 0.8801, 8:1 0.9000, 7:2 0.8881 (verified-from-source).
- Model structure: U-Net 0.5473 ... MEDIAR-Former + encoder init 0.8292 + head separation 0.8424 (verified-from-source).

## Limitations
- Author-stated: unlabelled data left for future work.
- Reviewer: contradictory description of unlabelled-data use between this paper and the challenge overview [[ma2024cellsegchallenge]]; ablations on 101 images.

## Relevance to CD / CV / HEX
We plan to use 8953 unlabelled-for-masks frames to improve masks (area CV r 0.38 zero-shot). MEDIAR is a negative datapoint: plain pseudo-labelling dropped F1 by 0.015 when labelled in-domain data existed. Our case differs (no in-domain masks at all, large domain gap, where [[reith2025selfadapt]] shows big gains), but the result argues for filtered, label-checked pseudo-labels rather than raw self-training. Flip TTA + small ensembles gave +0.4-0.7 F1 points, a cheap gain.

## What it suggests we try
Gate pseudo-masks by image-label agreement and mix in replay data from the 385 Topcon point images; add flip/rot90 TTA to the mask and point models at inference (well within 5 min/image on a T4).

Update 2026-11-09 (coordinator root-cause analysis): our CV/HEX error already equals an expert-centre reader plus ~150-cell sampling; better segmentation and label-guided segmenter self-training are closed for CV and HEX. Keep the line above as background, or for CD only.

## Related notes
- [[ma2024cellsegchallenge]] — challenge overview; says unlabelled data gave no notable gain for any top team
- [[reith2025selfadapt]] — unlabelled adaptation that does help when starting from a generalist
- [[pachitariu2025cellposesam]] — uses the NeurIPS 2022 data in training

## Sources
- https://ar5iv.labs.arxiv.org/html/2212.03465
- https://arxiv.org/abs/2212.03465
