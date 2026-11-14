---
id: zhang2020bigaug
type: paper
title: "Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation"
authors: [Ling Zhang, Xiaosong Wang, Dong Yang, Holger Roth, Andriy Myronenko, Daguang Xu, et al.]
year: 2020
venue: "IEEE Transactions on Medical Imaging"
doi: "10.1109/TMI.2020.2973595"
arxiv: "1906.03347"
pmcid: "PMC7393676"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7393676"
citation: "Zhang L, Wang X, Yang D, Roth H, Myronenko A, Xu D, et al. Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation. IEEE Trans Med Imaging. 2020;39(7):2531-2540. doi:10.1109/TMI.2020.2973595"
tier: core
group: t7-domain-shift
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "BigAug stacks nine augmentations (blur, sharpen, noise, brightness, contrast/gamma, intensity perturbation, rotation, scaling, deformation; each p=0.5) on single-source training and raises unseen-domain Dice from 49.8 to 80.0 on average, beating CycleGAN adaptation (63.5); sharpening, brightness and contrast mattered most for MRI."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Cheapest proven defence against acquisition shift without target data; directly applicable to our darker, vignetted 2023-24 source and to the HEX OOF-to-test gap, and costs nothing at inference."
code_release: unknown
weights_release: unknown
data_scale: "Prostate MRI (source MSD-P 26/6; unseen PROMISE12 50, NCI-ISBI13 60, ProstateX 98), left atrium MRI (MSD-H 16/4; ASC 100, MM-WHS 20), LV ultrasound (CETUS-A 8/2; CETUS-B/C 10 each); large-data 465-volume MRI source"
key_numbers:
  - "Unseen-domain average Dice: baseline 49.8, CycleGAN adaptation 63.5, top-4 augmentations 74.9, BigAug 80.0, supervised reference 91.5 (verified-from-source)"
  - "Source-domain Dice also rises: 89.1 baseline -> 91.6 BigAug (verified-from-source)"
  - "Single-transform ablation, unseen avg Dice: brightness 63.6, contrast 63.6, sharpening 62.9, blur 61.1, scaling 61.3, rotation 54.7, deformation 51.1 (verified-from-source)"
  - "Large data (465 volumes): unseen average 89.6 BigAug vs 87.5 baseline (verified-from-source)"
  - "Ranges: blur sigma 0.25-1.5, noise sigma 0.1-1.0, brightness +-0.1, gamma 0.5-1.0 or 1.0-4.5, scaling 0.4-1.6, rotation +-20 deg (verified-from-source)"
results:
  - {task: "3D segmentation, unseen domains", dataset: "7 unseen sets (MRI, US)", metric: "Dice (avg)", value: "80.0", comparator: "CycleGAN domain adaptation", comparator_value: "63.5", locator: "Table II", provenance: "verified-from-source"}
suggests_try: "Add BigAug-style appearance augmentation (gamma 0.5-2, brightness/contrast, radial vignetting field, blur/sharpen, noise) to the regression and detector training, and check the gap between old and new source MAPE plus OOF HEX; avoid geometric scaling, which changes CD."
tags: [domain-generalisation, data-augmentation, unseen-domain, scanner-shift, medical-segmentation]
search_keywords: [BigAug, deep stacked transformation, single-source domain generalization, augmentation unseen domain]
uses_dataset: [MSD prostate, PROMISE12, NCI-ISBI13, ProstateX, MSD heart, ASC, MM-WHS, CETUS]
uses_method: [stacked augmentation, AH-Net 3D segmentation]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "About 10 Dice points below supervised models at small data; ASC still 65.5; vendor labels for CETUS assigned visually", locator: "Results/Discussion"}
  - {stated_by: reviewer, note: "Segmentation only; for our regression, spatial scaling would alter the CD target and must be excluded or the label rescaled", locator: "n/a"}
---

# Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation

## TL;DR
Training on one source domain with nine stacked augmentations (image quality, appearance and spatial, each applied with probability 0.5) makes 3D medical segmentation models far more robust on unseen scanners/sites, outperforming a CycleGAN adaptation method that uses target data. With large training data, unseen-domain performance approaches supervised levels.

## Citation and link
- Zhang L, Wang X, Yang D, Roth H, Myronenko A, Xu D, et al. IEEE TMI. 2020;39(7):2531-2540. doi:10.1109/TMI.2020.2973595 (preprint arXiv 1906.03347, "When Unseen Domain Generalization is Unnecessary? Rethinking Data Augmentation")
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7393676

## Problem and motivation
Models trained on one scanner/site degrade on others; domain adaptation needs target data. Can augmentation alone generalise to unseen domains?

## Data
Prostate MRI, left atrium MRI, left ventricle ultrasound; sources small (8-26 training volumes); 7 unseen datasets; plus a 465-volume multi-centre MRI source.

## Method
Nine transforms, each p=0.5: Gaussian blur, unsharp sharpening, Gaussian noise; brightness shift, gamma contrast, intensity scale/shift; rotation, scaling, elastic deformation. 3D AH-Net segmenter.

## Results
- Unseen avg Dice: baseline 49.8; CycleGAN 63.5; top-4 aug 74.9; BigAug 80.0; supervised 91.5 (verified-from-source).
- Example: ASC (left atrium) 4.4 -> 65.5; CETUS-C 39.2 -> 81.3 (verified-from-source).
- Most useful single transforms: brightness and contrast (63.6 each), sharpening 62.9 (verified-from-source).

## Limitations
- Author-stated: gap to supervised at small data; computational cost of 3D augmentation.
- Reviewer: domain shift in MRI/US is larger than ours; magnitude of benefit for a mild brightness/vignetting shift between two specular sources is not known.

## Relevance to CD / CV / HEX
Our two sources differ in brightness and vignetting (CD error 7.11 old vs 5.50 new) and the hidden-test HEX error exceeded OOF by 0.40 MAPE. Appearance augmentation is the zero-cost lever: no test-time compute and no target data. Spatial scaling must be avoided for CD (density depends on pixel scale) unless labels are rescaled; HEX and CV are scale-invariant.

## What it suggests we try
Retrain the best regression model with strong appearance augmentation (gamma, brightness, contrast, synthetic vignetting, blur/sharpen, noise; p=0.5 each), report per-source MAPE and the old-new gap.

## Related notes
- [[karani2021tta]] — alternative: adapt an input normaliser at test time
- [[adachi2025ssa]] — test-time feature alignment for regression
- [[wang2019tta]] — test-time augmentation averaging

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7393676
