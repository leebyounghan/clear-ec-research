---
id: reith2025selfadapt
type: paper
title: "SelfAdapt: Unsupervised Domain Adaptation of Cell Segmentation Models"
authors: [Fabian H. Reith, Jannik Franzen, Dinesh R. Palli, J. Lorenz Rumberger, Dagmar Kainmueller]
year: 2025
venue: "ICCV 2025 Workshops (BioImage Computing, BIC); arXiv"
doi: "10.48550/arXiv.2508.11411"
arxiv: "2508.11411"
pmcid: ""
url: "https://arxiv.org/html/2508.11411v1"
citation: "Reith FH, Franzen J, Palli DR, Rumberger JL, Kainmueller D. SelfAdapt: Unsupervised Domain Adaptation of Cell Segmentation Models. Proc. ICCV Workshops (BIC). 2025. arXiv:2508.11411. doi:10.48550/arXiv.2508.11411"
tier: core
group: t4-cell-foundation
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: workshop
tldr: "Source-free, label-free adaptation of Cellpose (and other flow/distance-regression segmenters) to a new domain via EMA teacher-student consistency with TTA pseudo-labels, uncertainty masking of regression targets, L2-SP and label-free stopping; up to +29.6% relative AP0.5 over generalist Cellpose."
relevance_clearec: high
relevance_metrics: [CV, HEX, CD]
relevance_note: "A ready recipe (Cellpose extension, code open) to adapt a generalist segmenter to our 8953 unlabelled specular images; our image-level CD/CV/HEX labels can replace its weakest part, the label-free stopping rule."
code_release: open
weights_release: unknown
data_scale: "LiveCell and TissueNet official train/test splits (image counts not stated in the paper); 3 seeds"
key_numbers:
  - "Base generalist Cellpose AP0.5 0.415 -> 0.538 on LiveCell with SelfAdapt (best-checkpoint 'Test Max'), +29.6% relative (verified-from-source)"
  - "TissueNet cytoplasm 0.324 -> 0.395, TissueNet nuclei 0.603 -> 0.760 (Test Max) (verified-from-source)"
  - "Already fine-tuned Cellpose gains little: LiveCell 0.695 -> 0.705 at best; FN-stopping 0.693 (verified-from-source)"
  - "Label-free stopping captures only 52.8% (FN rule) / 71.5% (embedding rule) of the max gain on LiveCell vs 94-99% on TissueNet (verified-from-source)"
  - "Removing strong student augmentation collapses LiveCell to 0.451; removing L2-SP drops TissueNet nuclei to 0.642 (verified-from-source)"
  - "EMA alpha 0.99, 25,000 iterations, batch 8, AdamW; teacher = mean over flips and 90-degree rotations (verified-from-source)"
results:
  - {task: "instance segmentation", dataset: "LiveCell", metric: "AP0.5", value: "0.538", comparator: "generalist Cellpose cyto2", comparator_value: "0.415", locator: "Table 1", provenance: "verified-from-source"}
  - {task: "instance segmentation", dataset: "TissueNet nuclei", metric: "AP0.5", value: "0.760", comparator: "generalist Cellpose nuclei", comparator_value: "0.603", locator: "Table 1", provenance: "verified-from-source"}
suggests_try: "Run SelfAdapt-style EMA teacher-student on the 8953 unlabelled frames starting from Cellpose-SAM, but stop/select checkpoints by the correlation of mask-derived area CV and HEX with the image labels on a held-out fold (currently area CV r 0.38) instead of the label-free FN/embedding rules."
superseded: "2026-11-09: segmentation / self-training route closed for CV and HEX by the root-cause analysis; see body"
tags: [unsupervised-domain-adaptation, teacher-student, pseudo-labels, cellpose, source-free, early-stopping]
search_keywords: [SelfAdapt, ACTIS, L2-SP, mean teacher cell segmentation, label-free stopping criterion]
uses_dataset: [LiveCell, TissueNet]
uses_method: [mean-teacher, EMA, test-time-augmentation pseudo-labels, L2-SP, uncertainty masking]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "FN-rate stopping unreliable when the initial model over-segments; label-free stopping captures only part of the gain on LiveCell", locator: "Results / Discussion"}
  - {stated_by: author, note: "L2-SP strength tuned per model-dataset pair (1e-2 to 1e-5), so not hyperparameter-free", locator: "Methods"}
  - {stated_by: reviewer, note: "Evaluated only on fluorescence/phase-contrast benchmarks, base models cyto2/nuclei (not Cellpose-SAM); no packed-monolayer or specular data", locator: "Sec 4"}
---

# SelfAdapt: Unsupervised Domain Adaptation of Cell Segmentation Models

## TL;DR
SelfAdapt adapts a pretrained Cellpose (also applicable to StarDist, Mesmer) to a new image domain with no labels and no source data. An EMA teacher produces TTA-averaged pseudo-labels, a strongly augmented student is trained on them with per-pixel uncertainty masks (including regression outputs such as Cellpose flows), L2-SP keeps weights near the pretrained ones, and two label-free rules decide when to stop. Gains are large for generalist models (+22-30% relative AP0.5) and small for already fine-tuned models.

## Citation and link
- Reith FH, Franzen J, Palli DR, Rumberger JL, Kainmueller D. SelfAdapt. ICCV 2025 Workshops (BIC). arXiv:2508.11411.
- https://arxiv.org/html/2508.11411v1 ; code https://github.com/Kainmueller-Lab/self_adapt

## Problem and motivation
Generalist cell segmenters degrade on new microscopes/cell types; labelling a new domain is costly. Prior student-teacher consistency (ACTIS) handled only classification outputs and needed labelled validation data for stopping.

## Data
LiveCell (cytoplasm) and TissueNet (cytoplasm and nuclei), official training sets used unlabelled, test sets for evaluation. Image counts and cell types not stated in the paper. Three random seeds.

## Method
- Teacher: EMA of student (alpha 0.99); pseudo-label = average of predictions over flips and 90-degree rotations (inverse-transformed).
- Student: strong intensity/contrast changes, blur, Gaussian noise.
- Loss: cross-entropy on probability outputs + 0.5 x MSE on regression outputs (Cellpose flows), each masked by teacher uncertainty (std over TTA views; threshold = moving average of the 80th percentile).
- L2-SP penalty to pretrained weights (lambda 1e-2 to 1e-5 depending on model/dataset).
- Label-free stopping: (a) FN rate = share of instances found by the initial model that the current model misses, threshold 0.05; (b) mean bottleneck-embedding distance to the initial model, threshold 0.5; calibrated once on TissueNet nuclei.
- 25,000 iterations, batch 8, AdamW, warm-up + cosine. Wall-clock time not reported.

## Results
- LiveCell AP0.5: base 0.415; FN-stop 0.480; Emb-stop 0.503; best checkpoint 0.538 (verified-from-source).
- TissueNet cyto: 0.324 -> 0.394 / 0.391 / 0.395; nuclei 0.603 -> 0.759 / 0.754 / 0.760 (verified-from-source).
- Fine-tuned Cellpose: LiveCell 0.695 -> 0.693 / 0.704 / 0.705; TissueNet cyto 0.750 -> 0.752 / 0.755 / 0.763 (verified-from-source).
- Ablation: no student augmentation -> LiveCell 0.451, nuclei 0.615; no L2-SP -> nuclei 0.642 (verified-from-source).

## Limitations
- Author-stated: FN rule fails when the start model over-segments; prediction confidence rises monotonically while overfitting, so it is a bad stopping signal; L2-SP tuned per pair; thresholds calibrated on one dataset.
- Reviewer: no packed epithelial/endothelial monolayers, no Cellpose-SAM base; the adapted model is only as good as the teacher's systematic errors allow (merges that the teacher makes confidently are reinforced). Runtime not reported; 25k iterations at batch 8 should fit our 12 h budget but this is not shown.

## Relevance to CD / CV / HEX
Our zero-shot Cellpose-SAM/watershed masks give area CV r 0.38 with the label, and our point detector misses 15 % of cells (merges ruin area CV). SelfAdapt is the most direct published route to improve mask quality on our own 8953 frames without annotation. Crucially we have something the paper lacks: image-level CD/CV/HEX labels, which are a far better model-selection signal than its label-free stopping rules (which captured only 53-72 % of the gain on LiveCell).

## What it suggests we try
Teacher-student adapt Cellpose-SAM on the 8953 frames (teacher = flip/rot90 average, student = strong brightness/vignetting/blur/noise augmentation, L2-SP), and pick the checkpoint by held-out correlation of mask-derived CD/area CV/HEX with the labels; optionally drop pseudo-label images whose mask-derived CD disagrees with the label CD by more than ~10 %.

Update 2026-11-09 (coordinator root-cause analysis): our CV/HEX error already equals an expert-centre reader plus ~150-cell sampling; better segmentation and label-guided segmenter self-training are closed for CV and HEX. Keep the line above as background, or for CD only.

## Related notes
- [[pachitariu2025cellposesam]] — the base model we would adapt
- [[archit2025microsam]] — alternative base model with its own fine-tuning path
- [[lee2022mediar]] — another use of unlabelled target images (pseudo-labels) in cell segmentation

## Sources
- https://arxiv.org/html/2508.11411v1
- https://arxiv.org/abs/2508.11411
