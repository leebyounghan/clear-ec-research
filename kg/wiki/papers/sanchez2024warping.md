---
id: sanchez2024warping
type: paper
title: "Data augmentation via warping transforms for modeling natural variability in the corneal endothelium enhances semi-supervised segmentation"
authors: [Sergio Sanchez, Noelia Vallez, Gloria Bueno, Andres G. Marrugo]
year: 2024
venue: "PLoS ONE"
doi: "10.1371/journal.pone.0311849"
arxiv: ""
pmcid: "PMC11556704"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11556704/"
citation: "Sanchez S, Vallez N, Bueno G, Marrugo AG. Data augmentation via warping transforms for modeling natural variability in the corneal endothelium enhances semi-supervised segmentation. PLoS One. 2024;19(11):e0311849. doi:10.1371/journal.pone.0311849"
tier: candidate
group: t8-synthetic
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "Piecewise-affine warps on a Delaunay mesh of cell centroids create new image/mask pairs with altered cell geometry; used in Barlow Twins pretraining and fine-tuning, they raise Dice from 0.86 to 0.94 (DenseNet121) on 90 Topcon SP-3000P images, without any CD/CV/HEX evaluation."
relevance_clearec: medium
relevance_metrics: [CV, HEX]
relevance_note: "Shows centroid-mesh warping as a way to synthesise new cell geometries; for a regression target like CV/HEX such warps change the label, which matters for our augmentation choices."
code_release: unknown
weights_release: unknown
data_scale: "90 Topcon SP-3000P images (66 patients); 271 labelled + 1719 unlabelled 96 x 96 patches (597 / 4220 after warping)"
key_numbers:
  - "BT-DenseNet121: Dice 0.86 -> 0.94, mIoU 0.67 -> 0.86 from conventional augmentation (S1) to warping in both stages (S3) (verified-from-source)"
  - "BT-UNet: Dice 0.78 -> 0.85, mIoU 0.60 -> 0.72 (S1 -> S3) (verified-from-source)"
  - "Warp offset = R * d_min / s with R ~ U(-1, 1), s in 1-3; watershed used to repair warped masks (verified-from-source)"
  - "Test set only 25 labelled patches; no CD/CV/HEX validation (verified-from-source)"
  - "Dataset released on OSF (doi:10.17605/OSF.IO/75KMU) (verified-from-source)"
results:
  - {task: "cell segmentation", dataset: "Topcon SP-3000P patches (25 test)", metric: "Dice", value: "0.94 (BT-DenseNet121, S3)", comparator: "same model, conventional augmentation (S1)", comparator_value: "0.86", locator: "Table 2", provenance: verified-from-source}
suggests_try: "For the regression heads, avoid local elastic/piecewise warps as 'label-preserving' augmentation (they change area CV and neighbour counts); use only global zoom with a label update CD' = CD / s^2 (CV, HEX unchanged) to expand CD coverage, and reserve centroid-mesh warps for generating synthetic CV-shifted pretraining samples whose new CV is computed from the warped markers."
tags: [augmentation, warping, delaunay, semi-supervised, barlow-twins, topcon]
search_keywords: [warping augmentation corneal endothelium, semi-supervised endothelium segmentation, Barlow Twins endothelium]
uses_dataset: [Cartagena Topcon SP-3000P set (OSF 75KMU)]
uses_method: [piecewise affine warping, Delaunay triangulation, Barlow Twins, U-Net decoder with attention]
compares_with: []
extends: ["[[sierra2023sdm]]"]
has_limitation:
  - {stated_by: reviewer, note: "Tiny test set (25 patches), no CIs; abstract gains (+17.2 % mIoU, +4.8 % Dice) do not match Table 2; patient count mismatch (58 vs 66)", locator: "Table 1-2"}
  - {stated_by: reviewer, note: "No morphometric (CD/CV/HEX) evaluation", locator: "Results"}
---

# Data augmentation via warping transforms for the corneal endothelium

## TL;DR
Sanchez et al. warp specular images and masks with random piecewise-affine displacements on a Delaunay mesh of cell centroids, producing plausible new cell geometries. Adding these warps to Barlow Twins self-supervised pretraining and to fine-tuning lifts Dice from 0.86 to 0.94 for a DenseNet121 encoder on a small Topcon dataset.

## Citation and link
- Sanchez S, Vallez N, Bueno G, Marrugo AG. PLoS One. 2024;19(11):e0311849. doi:10.1371/journal.pone.0311849
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11556704/

## Problem and motivation
Labelled endothelium data are scarce; flips and rotations do not model natural variation in cell size and shape.

## Data
Topcon SP-3000P with Cell Count; 90 images (66 patients; healthy and Fuchs); 96 x 96 patches; dataset on OSF.

## Method
- Keypoints: mask centroids + edge points; Delaunay triangulation; random keypoint shifts scaled by the minimum neighbour distance; piecewise affine warp of image and mask; watershed repair.
- Barlow Twins pretraining (ResNet50/101, DenseNet121, ResNet101-ViT encoders) on unlabelled patches, then frozen-encoder fine-tuning with an attention decoder and BCE-Dice loss.

## Results
- Strategy 3 (warps in both stages) best for all encoders; DenseNet121 Dice 0.94, mIoU 0.86 (verified-from-source).
- No CD/CV/HEX numbers (verified-from-source).

## Limitations
- Author-stated: none explicit.
- Reviewer: small test set; inconsistent reported gains; morphometry untested.

## Relevance to CD / CV / HEX
Geometric warps that are fine for segmentation (mask warped too) are not label-preserving for image-level CV/HEX regression. Global isotropic zoom changes CD in a known way and leaves CV/HEX unchanged, so it is the safe geometric augmentation for CLEAR-EC.

## What it suggests we try
Check our current augmentation list: drop elastic/grid distortions from CV/HEX training, add zoom-with-CD-relabel (s in about 0.85-1.15, pixel scale fixed at 0.7716 um/px) to widen CD coverage, which may help the CD tails.

## Related notes
- [[sierra2023sdm]] — same Topcon dataset and group
- [[kucharski2023gan]] — synthetic data via mosaics + GAN
- [[marrugo2026review]] — review by the same group

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11556704/
