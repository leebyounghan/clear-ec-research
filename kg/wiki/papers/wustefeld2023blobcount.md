---
id: wustefeld2023blobcount
type: paper
title: "Learning to Segment Blob-like Objects by Image-Level Counting"
authors: [Konstantin Wüstefeld, Robin Ebbinghaus, Frank Weichert]
year: 2023
venue: "Applied Sciences"
doi: "10.3390/app132212219"
arxiv: ""
pmcid: ""
url: "https://mdpi.com/2076-3417/13/22/12219"
citation: "Wüstefeld K, Ebbinghaus R, Weichert F. Learning to Segment Blob-like Objects by Image-Level Counting. Appl Sci. 2023;13(22):12219. doi:10.3390/app132212219"
tier: candidate
group: t5-weak-supervision
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "A segmentation network with an added counting head is trained only on image-level object counts; a contrast loss and a morphological loss make the last layer produce high-contrast blobs, giving F1 comparable to full supervision on blob-like microscopy objects."
relevance_clearec: medium
relevance_metrics: [CD]
relevance_note: "Direct example of learning a per-object map from image-level counts in microscopy; our CD label is an image-level count proxy for the analysis region."
code_release: unknown
weights_release: unknown
data_scale: "not reported (full text not reachable)"
key_numbers:
  - "Supervision = image-level object counts only, via a counting head on a standard segmentation network (from-abstract)"
  - "Two extra losses (contrast loss, morphological loss) force a high-contrast, blob-like output before counting (from-abstract)"
  - "Reported F1 comparable to fully supervised training; exact values not reported in abstract (from-abstract)"
results: []
suggests_try: "Count-supervision only makes sense for CD here: our detector already has point supervision from 385 Topcon images, so use the CD-derived count in the analysis box as an extra loss on the detector heatmap (sum of blobs) rather than as the only label."
tags: [count-supervision, weak-supervision, blob-segmentation, microscopy]
search_keywords: [segmentation from image-level counts, counting head weak supervision]
uses_dataset: []
uses_method: [counting-head, contrast-loss, morphological-loss]
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "Full text blocked (HTTP 403); datasets, numbers and limitations not checked", locator: "n/a"}
  - {stated_by: reviewer, note: "Blob-like isolated objects; endothelial cells tile the plane with shared borders, so blob outputs map to cell centres, not cell masks", locator: "abstract"}
---

# Learning to Segment Blob-like Objects by Image-Level Counting

## TL;DR
The authors extend a segmentation network with a counting head and train it on per-image object counts only. Contrast and morphological losses shape the pre-count activation into clean blobs, which yields object detections/segmentations with F1 close to full supervision.

## Citation and link
- Wüstefeld K, Ebbinghaus R, Weichert F. Appl Sci. 2023;13(22):12219. doi:10.3390/app132212219.
- Read: abstract via search listings (MDPI and DOAJ pages returned HTTP 403).

## Problem and motivation
Pixel masks for microscopy blobs are costly; counts are cheap or already available.

## Data
Not reported in what was read.

## Method
Segmentation backbone + counting head; count loss on image-level counts; contrast loss and morphological loss penalise noisy, low-contrast outputs (from-abstract).

## Results
F1 comparable to fully supervised; numbers not reported in abstract (from-abstract).

## Limitations
- Author-stated: the abstract says limitations are discussed; not read.
- Reviewer: abstract only; blob assumption.

## Relevance to CD / CV / HEX
CD label = 1e6 / mean cell area, so for a known analysis region it implies a cell count; a count-supervised blob map is a centre detector, i.e. the same object as our point detector. Count supervision alone gives no handle on CV/HEX.

## What it suggests we try
Add a count-consistency term to the detector fine-tune on the 8953 labelled frames (see [[kervadec2019constrained]] for the bounded-penalty form) and check if it reduces the 15 % miss rate on the 25 expert frames.

## Related notes
- [[lempitsky2010counting]] — density-map counting, the continuous analogue.
- [[kervadec2019constrained]] — penalty form for an image-level constraint.

## Sources
- https://mdpi.com/2076-3417/13/22/12219 (403; abstract via search)
- https://doaj.org/article/bea2a026b0fc4c8394d2d01b38244a7d (403; abstract via search)
