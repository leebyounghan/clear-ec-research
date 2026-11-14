---
id: shumilin2023voronoi
type: paper
title: "A Method for Auto-Differentiation of the Voronoi Tessellation"
authors: [Sergei Shumilin, Alexander Ryabov, Serguei Barannikov, Evgeny Burnaev, Vladimir Vanovskii]
year: 2023
venue: "arXiv"
doi: "10.48550/arXiv.2312.16192"
arxiv: "2312.16192"
pmcid: ""
url: "https://arxiv.org/html/2312.16192v3"
citation: "Shumilin S, Ryabov A, Barannikov S, Burnaev E, Vanovskii V. A Method for Auto-Differentiation of the Voronoi Tessellation. arXiv:2312.16192 (v3, 2024)"
tier: candidate
group: t5-weak-supervision
status: reviewed
review_depth: light
reviewed_on: 2026-11-09
sources_read: [html, abstract]
evidence: preprint
tldr: "Makes 2-D Voronoi cell areas, edge lengths and vertex positions differentiable with respect to site coordinates (circumcentre formulas + shoelace areas in PyTorch); adjacency comes from a non-differentiable Delaunay step, so topology is fixed between retriangulations."
relevance_clearec: medium
relevance_metrics: [CV, HEX]
relevance_note: "The building block for a 'differentiable cell-graph head' (points -> Voronoi -> area CV) supervised by image-level CV, but it can only move points, not add or delete them, and neighbour counts (HEX) are not differentiable."
code_release: none
weights_release: none
data_scale: "toy experiments with 500-1000 points"
key_numbers:
  - "Differentiable: Voronoi vertices (circumcentres), edge lengths, cell areas via shoelace formula (verified-from-source)"
  - "Not differentiable: adjacency / neighbour relations from Delaunay triangulation (verified-from-source)"
  - "Bounded domains via differentiable Sutherland-Hodgman clipping; open border cells via ghost points at distance omega (verified-from-source)"
  - "Area-variance minimisation with 1000 points, 1400 Adam steps at lr 0.001, variance converges almost to zero (qualitative only) (verified-from-source)"
  - "No wall-clock timings or accuracy benchmarks reported; no code link in paper (verified-from-source)"
results: []
suggests_try: "Not worth building in 2.5 days: our dominant error is 15 % missed / 16.5 % false points, which position gradients cannot fix, and HEX needs neighbour counts that are non-differentiable here; a cheaper route is fitting a calibration from non-differentiable Voronoi/neighbour-distance CV to the CV label."
tags: [differentiable-geometry, Voronoi, autodiff, cell-graph]
search_keywords: [differentiable Voronoi, auto-differentiation Voronoi tessellation, soft Voronoi]
uses_dataset: []
uses_method: [differentiable-Voronoi, Delaunay, shoelace-area]
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Topology fixed between retriangulations; vertices can leave the bounded domain; ghost-point distance is a hyperparameter", locator: "Limitations"}
  - {stated_by: reviewer, note: "Qualitative toy experiments only; no learning-from-labels experiment", locator: "Experiments"}
---

# A Method for Auto-Differentiation of the Voronoi Tessellation

## TL;DR
The paper writes 2-D Voronoi geometry as smooth functions of site coordinates (vertices = circumcentres of Delaunay triangles, areas by the shoelace formula) so any autograd loss on areas or edges can push gradients back to point positions. Demonstrated on toy area-variance and facility-placement objectives.

## Citation and link
- Shumilin S, Ryabov A, Barannikov S, Burnaev E, Vanovskii V. arXiv:2312.16192 (v3 Dec 2024).
- Read: https://arxiv.org/html/2312.16192v3 ; https://arxiv.org/abs/2312.16192

## Problem and motivation
Pipelines that contain a Voronoi step (spatial partitioning, tissue models) cannot be trained end-to-end because the tessellation is a discrete algorithm.

## Data
Synthetic point sets (500-1000 points) in the unit square (verified-from-source).

## Method
Delaunay triangulation outside the graph; circumcentre formulas (Eqs. 1-3) inside the graph; border handling by ghost points (unbounded) or differentiable Sutherland-Hodgman clipping (convex bounded domain); periodic retriangulation; PyTorch (verified-from-source).

## Results
- Area-variance minimisation converges to near-equal areas (histogram at the theoretical 0.001 per cell); qualitative (verified-from-source).
- Facility placement example; no final error value reported (verified-from-source).

## Limitations
- Author-stated: topology changes not differentiated; vertices may leave the domain; omega hyperparameter.
- Reviewer: no runtime, no comparison, no supervised-learning use case.

## Relevance to CD / CV / HEX
A head that turns detected points into Voronoi areas and computes area CV could be trained against the image-level CV label. But (a) device CV is about twice the Voronoi-area CV of expert markers, so a scale/calibration is needed anyway; (b) the detector's misses (15 %) and false points (16.5 %) are discrete errors that gradients on positions cannot correct; (c) HEX depends on neighbour counts, which are non-differentiable here.

## What it suggests we try
Nothing actionable in the time left; keep Voronoi/neighbour statistics as non-differentiable features fed to the regression head instead.

## Related notes
- [[kervadec2019constrained]] — differentiable statistic of a soft mask, the linear case that does work.
- [[qu2020pointnuclei]] — Voronoi labels as pseudo-supervision instead of a differentiable layer.

## Sources
- https://arxiv.org/html/2312.16192v3
- https://arxiv.org/abs/2312.16192
