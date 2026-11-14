---
id: doughty2018regional
type: paper
title: "On the regional variability of averaged cell area estimates for the human corneal endothelium in relation to the extent of polymegethism"
authors: [Michael J. Doughty]
year: 2018
venue: "International Ophthalmology"
doi: "10.1007/s10792-017-0765-2"
arxiv: ""
pmcid: "PMC6267656"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6267656"
citation: "Doughty MJ. On the regional variability of averaged cell area estimates for the human corneal endothelium in relation to the extent of polymegethism. Int Ophthalmol. 2018;38(6):2537-2546 (online 2017). doi:10.1007/s10792-017-0765-2"
tier: core
group: t1-endothelium-segmentation
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "Hand-traced areas of 200 cells in each of 80 Topcon SP-3000P images show that mean cell area is stable across 20-cell blocks, but COV of cell area becomes unreliable with 50 or fewer cells in polymegethous endothelium (can differ by +/-10 % or more); at least 75-100 contiguous cells are recommended."
relevance_clearec: high
relevance_metrics: [CD, CV]
relevance_note: "Quantifies how much CV (and much less mean area/CD) depends on which and how many cells are sampled; directly bears on label noise from the ~150-cell technician-chosen region and on the expected CV floor."
code_release: none
weights_release: none
data_scale: "80 central Topcon SP-3000P images (4 groups of 20 by global COV), 200 hand-traced cells each (16,000 cells)"
key_numbers:
  - "Pooled 16,000 cells: mean area 421 um2; 20-cell block means within +/-14 um2 of the overall mean; SD of block means about 70 um2 (verified-from-source)"
  - "With <= 50 cells in moderate-to-marked polymegethism, global COV could differ by +/-10 % or more (verified-from-source)"
  - "Recommended: 75-100 contiguous cells per image, at least about 100 for reliable estimates (verified-from-source)"
  - "Manually traced global COV spans <30 % (uniform) to >50-60 % (marked) on Topcon SP-3000P images (verified-from-source)"
  - "Uniform endothelium (COV < 30 %): COV variability drops beyond 120 cells (6 blocks) (verified-from-source)"
results:
  - {task: "COV stability vs number of sampled cells", dataset: "80 Topcon SP-3000P images, 200 traced cells each", metric: "COV variability across sequential 20-cell blocks", value: "+/-10 % or more at <= 50 cells (moderate-marked polymegethism)", comparator: "200-cell global COV", comparator_value: "reference", locator: "Results/Discussion", provenance: verified-from-source}
suggests_try: "Model label noise as heteroscedastic: CV label noise grows with the label's own CV (polymegethous frames vary more by region), so weight CV loss by 1/(expected sampling variance) or use a noise-aware loss, and do not chase the outer CV deciles with heavy per-sample weighting beyond what this sampling noise allows."
tags: [polymegethism, sampling, cell-count, planimetry, topcon, classical]
search_keywords: [number of cells coefficient of variation endothelium, regional variability cell area, polymegethism sampling]
uses_dataset: []
uses_method: [manual planimetry, sequential block subsampling]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Single observer; analysis possibly statistical overkill", locator: "Discussion"}
  - {stated_by: reviewer, note: "Sequential contiguous blocks, not random sampling; groups defined by the outcome variable; abstract and results disagree on whether >100 cells helps in marked polymegethism", locator: "Methods/Abstract"}
---

# Regional variability of averaged cell area estimates in relation to polymegethism

## TL;DR
Doughty traces 200 cells per image in 80 Topcon SP-3000P endothelial images and recomputes mean area and COV from growing sets of contiguous 20-cell blocks. Mean area is stable, but COV is unstable when 50 or fewer cells are used in polymegethous endothelium (differences of 10 % or more), and regional differences inside one image can be substantial. He recommends 75-100 contiguous cells.

## Citation and link
- Doughty MJ. Int Ophthalmol. 2018;38(6):2537-2546. doi:10.1007/s10792-017-0765-2 (open access, CC BY 4.0)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6267656

## Problem and motivation
How many cells are needed for stable mean area and COV, and does it depend on polymegethism?

## Data
Topcon SP-3000P (some SP-2000P); 80 central images grouped by global COV (< 30 %, 31-40 %, 41-50 %, > 50 %); 200 cells hand-outlined per image on prints; planimetry repeatability +/-2 %.

## Method
Ten sequential 20-cell blocks per image, pooled cumulatively from 60 to 200 cells; variability of mean area and COV compared with Friedman and t-tests.

## Results
- Mean area 421 um2 overall; block means within +/-14 um2 (verified-from-source).
- COV unstable at <= 50 cells in moderate/marked polymegethism, +/-10 % or more (verified-from-source).
- Uniform endothelium: COV variability reduced beyond 120 cells (verified-from-source).
- Recommendation 75-100 contiguous cells (verified-from-source).

## Limitations
- Author-stated: single observer.
- Reviewer: contiguous blocks confound sampling size with position; internal inconsistency on > 100 cells.

## Relevance to CD / CV / HEX
CLEAR-EC labels come from about 150 cells in a technician-chosen region of 17 % of the frame. Doughty shows that (1) mean area (hence CD) is robust to which cells are chosen, consistent with our low CD error, and (2) COV depends on region and count, especially when COV is high, so the label CV in the upper deciles is intrinsically noisier. Manual traced COV on Topcon images spans 30-60 %, which is the same range as our device labels (0.29-0.50), while Voronoi-from-centers CV is about half; this suggests Voronoi tessellation compresses area variability rather than the device inflating it.

## What it suggests we try
(1) Use heteroscedastic weighting for CV (lower weight where label CV is high); (2) for CV features from detected centers, compute traced-area proxies (watershed regions) rather than Voronoi areas, since traced-area COV ranges match label ranges; (3) compute CV from several random ~150-cell subsets of the frame and feed the mean and spread as features.

## Related notes
- [[viguerasguillen2020dsaek]] — CV overestimated below ~50 detected cells
- [[viguerasguillen2022feedbackattn]] — HEX underestimated below 100 cells
- [[scarpa2016morphometry]] — inter-expert CV floor

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6267656
