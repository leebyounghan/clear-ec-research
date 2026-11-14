---
id: kucharski2021cnnwatershed
type: paper
title: "CNN-watershed: A watershed transform with predicted markers for corneal endothelium image segmentation"
authors: [Adrian Kucharski, Anna Fabijanska]
year: 2021
venue: "Biomedical Signal Processing and Control"
doi: "10.1016/j.bspc.2021.102805"
arxiv: ""
pmcid: ""
url: "https://api.openalex.org/works?search=CNN-watershed%20predicted%20markers%20corneal%20endothelium"
citation: "Kucharski A, Fabijanska A. CNN-watershed: A watershed transform with predicted markers for corneal endothelium image segmentation. Biomed Signal Process Control. 2021;68:102805. doi:10.1016/j.bspc.2021.102805"
tier: candidate
group: t1-endothelium-segmentation
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [abstract]
evidence: peer-reviewed
tldr: "A CNN predicts cell markers (one blob per cell) and edges, and a marker-controlled watershed turns them into a tessellation; a later review lists Dice 0.868 and R2 0.929 on 119 images (animal, healthy, Fuchs)."
relevance_clearec: medium
relevance_metrics: [CD, CV, HEX]
relevance_note: "Marker (cell-center) prediction + watershed is the closest classical analogue to our point-detector pipeline; it turns centers into areas without assuming Voronoi geometry."
code_release: unknown
weights_release: unknown
data_scale: "119 images (animal, healthy, FECD) per the Marrugo 2026 review; details not read"
key_numbers:
  - "Dice 0.868 and Pearson R2 0.929 (variable not specified), as tabulated in Marrugo et al. 2026 review Table 1 (verified-from-source)"
  - "119 images, animal + healthy + FECD, per the same review table (verified-from-source)"
  - "Abstract and full text not accessible (closed access; abstract field empty in OpenAlex and Semantic Scholar) (from-abstract)"
results: []
suggests_try: "Instead of Voronoi areas from detected centers (device CV is about twice Voronoi CV), grow cells from our center detections with a marker-controlled watershed on an edge/intensity map and compute area CV from those regions; check on the 25 expert-marker frames whether watershed-area CV tracks device CV better than Voronoi-area CV."
superseded: "2026-11-09: segmentation / self-training route closed for CV and HEX by the root-cause analysis; see body"
tags: [watershed, marker-prediction, u-net, classical-hybrid]
search_keywords: [CNN-watershed, predicted markers watershed endothelium, marker-controlled watershed corneal]
uses_dataset: []
uses_method: [U-Net, marker-controlled watershed]
compares_with: []
extends: ["[[fabijanska2018unet]]"]
has_limitation:
  - {stated_by: reviewer, note: "Closed access; only metadata and a secondary review table were read, so method details and numbers are unverified at source", locator: "n/a"}
---

# CNN-watershed: watershed with predicted markers for corneal endothelium segmentation

## TL;DR
Kucharski and Fabijanska replace hand-crafted watershed markers with CNN-predicted markers so each endothelial cell is seeded once, then run a watershed to obtain cell borders. A 2026 review tabulates Dice 0.868 and R2 0.929 on 119 images including animal, healthy and Fuchs corneas.

## Citation and link
- Kucharski A, Fabijanska A. Biomed Signal Process Control. 2021;68:102805. doi:10.1016/j.bspc.2021.102805
- Metadata via OpenAlex; numbers via [[marrugo2026review]] Table 1.

## Problem and motivation
Marker-driven watershed is the standard classical method, but markers from intensity minima over- or under-seed cells in low-contrast images.

## Data
119 images (animal, healthy, FECD) per the review; probably the Lodz group's public endothelium sets (not verified).

## Method
CNN predicts markers (and possibly edge probability); marker-controlled watershed on the image gradient gives the tessellation. Indexing keyphrases mention region merging and multiscale processing (not verified).

## Results
- Dice 0.868; R2 0.929 (review Table 1) (verified-from-source).
- CV/HEX errors: not reported in what was read.

## Limitations
- Author-stated: not available.
- Reviewer: unverified at source; small datasets.

## Relevance to CD / CV / HEX
Our detector already provides markers; the missing step is growing real cell regions. Since the device CV is about twice the Voronoi-area CV of expert markers, the device likely measures traced-border areas (which include border irregularity and merged-cell effects), so watershed areas may be closer to the device definition than Voronoi areas.

## What it suggests we try
On the 25 expert-marker frames, compute area CV from (a) Voronoi cells and (b) watershed regions seeded by the same markers on the image; keep whichever correlates better with device CV and use it as a feature.

Update 2026-11-09 (coordinator root-cause analysis): our CV/HEX error already equals an expert-centre reader plus ~150-cell sampling; better segmentation and label-guided segmenter self-training are closed for CV and HEX. Keep the line above as background, or for CD only.

## Related notes
- [[fabijanska2018unet]] — U-Net edge segmentation by the same lab
- [[kucharski2023gan]] — synthetic training data from the same lab
- [[marrugo2026review]] — source of the tabulated numbers

## Sources
- https://api.openalex.org/works?search=CNN-watershed%20predicted%20markers%20corneal%20endothelium
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13038023/
