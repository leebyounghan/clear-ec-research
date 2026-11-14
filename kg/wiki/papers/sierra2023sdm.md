---
id: sierra2023sdm
type: paper
title: "Corneal endothelium assessment in specular microscopy images with Fuchs' dystrophy via deep regression of signed distance maps"
authors: [Juan S. Sierra, Jesus Pineda, Daniela Rueda, Alejandro Tello, Angelica M. Prada, Virgilio Galvis, et al.]
year: 2023
venue: "Biomedical Optics Express"
doi: "10.1364/BOE.477495"
arxiv: "2210.07102"
pmcid: "PMC9842012"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC9842012"
citation: "Sierra JS, Pineda J, Rueda D, Tello A, Prada AM, Galvis V, Volpe G, Millan MS, Romero LA, Marrugo AG. Corneal endothelium assessment in specular microscopy images with Fuchs' dystrophy via deep regression of signed distance maps. Biomed Opt Express. 2023;14(1):335-351. doi:10.1364/BOE.477495"
tier: core
group: t1-endothelium-segmentation
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc, abstract]
evidence: peer-reviewed
tldr: "A UNet regresses one signed distance map (cells positive, guttae negative) with MAE loss; thresholding plus watershed gives cells, and on 23 Topcon SP-3000P test images CV differs from manual by +2.7 points on average, while the Topcon CellCount software differs by +29.9 CV points with a bias that grows with CV."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Shows device software CV can be far above manual area CV (mean +29.9 points, rising with CV), consistent with our device CV being about twice the Voronoi CV; distance-map regression is a cheap way to avoid merged cells."
code_release: open
weights_release: open
data_scale: "90 Topcon SP-3000P images from 66 patients (42 healthy, 48 Fuchs corneas); 57/10/23 split"
key_numbers:
  - "UNet-dm vs manual (23 test images): CD mean diff -41.9 cells/mm2 (95 % LoA -306.2 to 222.5); HEX +0.8 (+/-10.4); CV +2.7 (-10.2 to 15.6) (verified-from-source)"
  - "Topcon CellCount vs manual: CD +312.4 cells/mm2; HEX +1.5 (+/-11.2); CV +29.9 (LoA -60.6 to 120.4) with error rising as CV rises (verified-from-source)"
  - "Topcon software segments guttae as large cells or several small cells (verified-from-source)"
  - "Pixel accuracy: UNet-dm 83.79 % vs 3-class UNet 83.06 %; UNet classification produced merged cells and underestimated CD and HEX in examples (verified-from-source)"
  - "Training about 33 min on one Tesla T4 (verified-from-source)"
results:
  - {task: "CV estimation", dataset: "23 Topcon SP-3000P test images", metric: "Bland-Altman mean difference (CV points)", value: "+2.7 (UNet-dm)", comparator: "Topcon CellCount", comparator_value: "+29.9", locator: "Results, Fig. 9", provenance: verified-from-source}
  - {task: "CD estimation", dataset: "23 test images", metric: "Bland-Altman mean difference (cells/mm2)", value: "-41.9", comparator: "Topcon CellCount", comparator_value: "+312.4", locator: "Results, Fig. 9", provenance: verified-from-source}
suggests_try: "Train a signed/normalised distance-to-boundary regression head (MAE loss) using pseudo-masks grown from the Topcon point set (dot-seeded watershed), then derive cells by threshold + watershed; distance-map peaks double as robust cell centers that merge less than our current detector."
superseded: "2026-11-09: segmentation / self-training route closed for CV and HEX by the root-cause analysis; see body"
tags: [signed-distance-map, regression, guttae, fuchs, topcon, watershed, deeptrack]
search_keywords: [signed distance map endothelium, UNet-dm, guttae area ratio GAR, Topcon CellCount error]
uses_dataset: [Cartagena Topcon SP-3000P Fuchs set]
uses_method: [UNet, signed distance map regression, watershed]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Small set (90 images, 23 test), single device/site, retrospective; peripheral non-uniform illumination produced false guttae", locator: "Discussion"}
  - {stated_by: reviewer, note: "Manual references were made by editing the Topcon segmentation, which may bias toward it; augmentation count (148 samples) inconsistent with 57 training images", locator: "Methods"}
---

# Corneal endothelium assessment with Fuchs' dystrophy via deep regression of signed distance maps

## TL;DR
Sierra et al. turn segmentation into regression: a UNet predicts a signed distance map that is positive inside cells and negative inside guttae. Thresholding and watershed then give cells and guttae, and CD, CV, HEX and a new guttae-area ratio. On 23 test images the method agrees with manual references far better than the Topcon CellCount software, whose CV is on average 29.9 points too high.

## Citation and link
- Sierra JS, et al. Biomed Opt Express. 2023;14(1):335-351. doi:10.1364/BOE.477495 (arXiv:2210.07102)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9842012 ; code and weights on GitHub, archived on Zenodo (doi:10.5281/zenodo.7378507)

## Problem and motivation
In Fuchs dystrophy, guttae hide cells and vendor software counts guttae as cells; pixel classification UNets merge touching cells.

## Data
Topcon SP-3000P, 640 x 480 exports, 90 images from 66 patients (42 healthy, 48 dystrophic); physicians edited the Topcon segmentation into cell and gutta masks; 57 train / 10 val / 23 test; 96 x 96 patches.

## Method
- Target DI = distance transform of cell mask minus distance transform of gutta mask.
- 5-level UNet with instance norm, Leaky ReLU, ResNet bottom; MAE loss; Adam; flips/rotations; DeepTrack.
- Guttae = DI < 0; cells = DI > 0.2; watershed on the union; border cells removed.
- CD = cells / area; CV = std/mean cell area; HEX = share of six-neighbour cells; GAR = guttae area share.

## Results
- UNet-dm vs manual: CD -41.9 cells/mm2, MCA +14.8 um2, HEX +0.8, CV +2.7 (verified-from-source).
- Topcon CellCount vs manual: CD +312.4, MCA +54.3, HEX +1.5, CV +29.9 with upward trend (verified-from-source).
- Distance-map regression converges faster than 3-class UNet on CD/HEX/CV error curves (verified-from-source, values only in Fig. 7).

## Limitations
- Author-stated: small, single-site; illumination artefacts; manual references may contain merges.
- Reviewer: no r or MAPE; CV/HEX means not reported, so relative size of the +29.9 CV bias cannot be computed.

## Relevance to CD / CV / HEX
The striking result for CLEAR-EC is the Topcon software CV bias (+29.9 points on average, larger at high CV, with wide limits). It independently supports our observation that device CV is about twice the Voronoi area CV of expert markers: device CV is its own quantity, inflated by guttae or noise segmented as small/large cells. HEX bias of the device is small (+1.5), matching our finding that HEX labels track expert six-neighbour share within 1-2 %.

## What it suggests we try
Model device CV as "true CV + segmentation-noise inflation": add image-quality features (dark-spot/guttae area fraction, blur, local contrast) to the CV head, since the device's CV error grows with these; and try a distance-map regression head trained on dot-seeded pseudo-masks from the external point set.

Update 2026-11-09 (coordinator root-cause analysis): our CV/HEX error already equals an expert-centre reader plus ~150-cell sampling; better segmentation and label-guided segmenter self-training are closed for CV and HEX. Keep the line above as background, or for CD only.

## Related notes
- [[viguerasguillen2022feedbackattn]] — attention-based guttae handling on Topcon SP-1P
- [[marrugo2026review]] — review by the same group
- Prada et al. 2024 (Cornea) — follow-up clinical study on estimated ECD and GAR% (no note in this KG)

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9842012
