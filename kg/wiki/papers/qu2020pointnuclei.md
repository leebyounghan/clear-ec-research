---
id: qu2020pointnuclei
type: paper
title: "Weakly Supervised Deep Nuclei Segmentation Using Partial Points Annotation in Histopathology Images"
authors: [Hui Qu, Pengxiang Wu, Qiaoying Huang, Jingru Yi, Zhennan Yan, Kang Li, et al.]
year: 2020
venue: "IEEE Transactions on Medical Imaging"
doi: "10.1109/TMI.2020.3002244"
arxiv: "2007.05448"
pmcid: ""
url: "https://ar5iv.labs.arxiv.org/html/2007.05448"
citation: "Qu H, Wu P, Huang Q, Yi J, Yan Z, Li K, et al. Weakly Supervised Deep Nuclei Segmentation Using Partial Points Annotation in Histopathology Images. IEEE Trans Med Imaging. 2020;39(11):3655-3666. doi:10.1109/TMI.2020.3002244"
tier: core
group: t5-weak-supervision
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: peer-reviewed
tldr: "From only 5-50 % of nucleus centre points, a self-trained detector finds the rest; Voronoi labels (edges = background, dilated points = foreground) plus k-means colour cluster labels and a CRF loss train a segmentation net that reaches F1 0.77 vs 0.88 fully supervised on lung cancer images."
relevance_clearec: high
relevance_metrics: [CV, HEX, CD]
relevance_note: "The standard recipe for turning cell-centre points into masks without manual masks; our Topcon points and detector give the seeds, and endothelium is closer to a Voronoi tessellation than nuclei are."
code_release: open
weights_release: unknown
data_scale: "Lung Cancer 40 images 900x900 (24/8/8 split, 24,401 nuclei); Multi-Organ 30 images 1000x1000 (12/4/14)"
key_numbers:
  - "LC segmentation F1/Dice/AJI: full 0.8771/0.8521/0.6979; all GT points 0.8143/0.8021/0.6497; 10 % points 0.7700/0.7574/0.5754 (verified-from-source)"
  - "MO: full 0.8100/0.6763/0.3919; all GT points 0.7716/0.7242/0.5174; 10 % points 0.7490/0.7033/0.5031 (verified-from-source)"
  - "Detection F1 with 10 % points (self-training + background propagation): LC 0.8879 vs 0.8950 full; MO 0.8282 vs 0.8541 (verified-from-source)"
  - "Voronoi label: edges = background, points dilated by radius-2 disk = nucleus, rest ignored; cluster label: k-means k = 3 on distance + RGB (verified-from-source)"
  - "Voronoi-only better on object-level metrics, cluster-only better on pixel-level; alpha = 0.5 mix used (values only in figure) (verified-from-source)"
results:
  - {task: "nuclei segmentation", dataset: "Lung Cancer", metric: "F1 (object)", value: "0.7700 (10 % points)", comparator: "fully supervised", comparator_value: "0.8771", locator: "Table III", provenance: "verified-from-source"}
suggests_try: "Generate Voronoi pseudo-labels from detector points on all 8953 frames (edges = border, dilated centres = cell) plus intensity cluster labels (dark borders), train a border-segmentation net, and compute area CV from point-seeded watershed on its output; test on the 25 expert frames whether this area CV beats the zero-shot mask r 0.38."
superseded: "2026-11-09: segmentation / self-training route closed for CV and HEX by the root-cause analysis; see body"
tags: [point-supervision, Voronoi-labels, nuclei-segmentation, self-training, CRF-loss]
search_keywords: [Voronoi pseudo labels nuclei, partial points annotation, weakly supervised nuclei segmentation]
uses_dataset: [Lung Cancer (Qu), MoNuSeg Multi-Organ]
uses_method: [Voronoi-labels, k-means-cluster-labels, dense-CRF-loss, self-training]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Voronoi edges between touching nuclei are not true background; Voronoi labels lack boundary/shape detail, hence cluster labels", locator: "Method"}
  - {stated_by: reviewer, note: "Histopathology nuclei are sparse blobs on background; endothelial cells tile the plane, so 'background' = thin borders only", locator: "Method"}
---

# Weakly Supervised Deep Nuclei Segmentation Using Partial Points Annotation in Histopathology Images

## TL;DR
Qu et al. segment nuclei with only a fraction of centre points annotated. A detector is self-trained to recover missing points, then Voronoi labels and colour-cluster labels derived from the points train a segmentation network with a CRF loss. Performance stays within about 0.1 F1 of full supervision.

## Citation and link
- Qu H, et al. IEEE Trans Med Imaging. 2020;39(11):3655-3666. doi:10.1109/TMI.2020.3002244.
- Read: https://ar5iv.labs.arxiv.org/html/2007.05448

## Problem and motivation
Full nuclei masks are expensive; points are cheap, and even partial points (some nuclei per image) should suffice.

## Data
Lung Cancer: 40 H&E images 900x900, 8 cases, 24/8/8, 24,401 nuclei. Multi-Organ (Kumar): 30 images 1000x1000, 7 organs, 12/4/14 (verified-from-source).

## Method
- Detection: extended Gaussian masks; self-training with background propagation (background where prob < 0.1).
- Voronoi labels: detected points as seeds, Voronoi edges = background, radius-2 dilated points = nucleus.
- Cluster labels: k-means (k = 3) on clipped distance + RGB, refined morphologically within each Voronoi cell.
- Segmentation loss: weighted CE on both labels (alpha = 0.5), then dense-CRF loss fine-tuning (20 epochs) (verified-from-source).

## Results
- LC F1/Dice/AJI at 5/10/25/50 % points: F1 0.7612/0.7700/0.7768/0.7819 vs full 0.8771 (verified-from-source).
- MO: F1 0.7540/0.7490/0.7511/0.7566 vs full 0.8100 (full model not post-processed, so its Dice/AJI are low) (verified-from-source).
- Detection F1 (10 % points): 0.8879 (LC), 0.8282 (MO) (verified-from-source).

## Limitations
- Author-stated: Voronoi edges between touching nuclei are approximations; Voronoi labels carry no shape detail.
- Reviewer: evaluated on segmentation metrics, not on morphometric statistics (area variance), which are more sensitive to merges and splits.

## Relevance to CD / CV / HEX
Our CV fact matters here: device CV is about twice the Voronoi-area CV of expert markers, so pure Voronoi cells from centres understate true area variability; Qu's cluster labels (image evidence) are what would recover real borders. Our detector's 15 % misses merge cells, which in a Voronoi/watershed pipeline inflates area CV; their self-training-with-background-propagation step is a direct recipe for reducing misses without manual labels.

## What it suggests we try
(1) Self-train the point detector on the 8953 unlabelled-for-points frames (pseudo-points with confidence threshold, background propagation), check miss/false rates on the 25 expert frames. (2) Voronoi + dark-border cluster pseudo-labels -> border net -> seeded watershed -> area CV; compare its r with the CV label against the 0.38 zero-shot baseline.

Update 2026-11-09 (coordinator root-cause analysis): our CV/HEX error already equals an expert-centre reader plus ~150-cell sampling; better segmentation and label-guided segmenter self-training are closed for CV and HEX. Keep the line above as background, or for CD only.

## Related notes
- [[kervadec2019constrained]] — adding image-level CD as a size constraint to such pseudo-label training.
- [[lempitsky2010counting]] — density-map alternative for counting.

## Sources
- https://ar5iv.labs.arxiv.org/html/2007.05448
- https://unpaywall.org/10.1109%2FTMI.2020.3002244 (via search, DOI record)
