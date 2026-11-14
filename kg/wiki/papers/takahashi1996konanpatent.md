---
id: takahashi1996konanpatent
type: paper
title: "Method for computing morphology of cornea endothelium cells (US5523213A)"
authors: [Yoko Takahashi, Tatsuya Kasahara]
year: 1996
venue: "US Patent 5,523,213 (assignee Konan Inc.)"
doi: ""
arxiv: ""
pmcid: ""
url: "https://patents.google.com/patent/US5523213A/en"
citation: "Takahashi Y, Kasahara T. Method for computing morphology of cornea endothelium cells. US Patent 5,523,213 A, assigned to Konan Inc.; priority JP 5-089244 (1993-03-23); granted 1996-06-04."
tier: core
group: t2-device-morphometry
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [html]
evidence: patent
tldr: "Konan's patented centre method builds each cell polygon from operator-clicked centres with a power-WEIGHTED bisector (cut point placed by the ratio of the two cells' average neighbour distances raised to the 2nd-4th power), not a plain Voronoi midpoint; CD = 1e6 / mean area, CV = SD/mean area, HEX = share of six-vertex polygons."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "The most direct description of how a Konan device turns cell centres into the areas behind CV and HEX; the weighting amplifies area spread versus Voronoi and plausibly explains our 'device CV about 2x Voronoi CV' fact."
code_release: none
weights_release: none
data_scale: "not applicable (patent; no dataset)"
key_numbers:
  - "Neighbour search radius = 3 x nearest-centre distance Lmin (claimed range 2.5-3.5 x) (verified-from-source)"
  - "Neighbour pruning: drop the farther of two successive neighbours if their angle at the centre < ~30 deg (claimed 20-40 deg); keep middle neighbour if successive-triplet angle > ~128 deg (claimed 120-140 deg) (verified-from-source)"
  - "Cell flagged peripheral (excluded from vertex construction) if a gap between successive neighbours > ~100 deg (claimed 90-110 deg) (verified-from-source)"
  - "Bisector cut point divides the centre-to-neighbour segment in the ratio of the two cells' average neighbour distances raised to the 2nd-4th power (summary: 'square to fourth power'; claim 6: proportional) (verified-from-source)"
  - "CD = 1,000,000 / mean cell area; CV = SD / mean area; hexagonal frequency = n hexagonal cells / N (verified-from-source)"
  - "Operator clicks every centre ('200 times for 200 cells') (verified-from-source)"
results: []
suggests_try: "Re-implement this weighted-bisector polygon construction (3xLmin neighbours, angle pruning, peripheral exclusion, cut ratio d_a^k : d_b^k) on the 25 expert-marker frames, fit the single exponent k (k=0 is Voronoi) to device CV/HEX, and check whether CV r rises above 0.89 and the 2x scale factor disappears; then feed the same emulator our detector points as a 'device-emulated CV/HEX' feature for the regressor."
tags: [konan, center-method, device-algorithm, weighted-voronoi, cv, hexagonality, patent]
search_keywords: ["Konan center method algorithm", "corneal endothelium cell boundary estimation from center points", "JP3265044 center method", "weighted bisector endothelial cell area"]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "A patent, not a validation study: no accuracy numbers; the shipping CellChek software (centre, flex-centre, auto-centre) may use later variants or different parameters.", locator: "whole document"}
  - {stated_by: reviewer, note: "Equations 2-4 were not legible in the Google Patents text; the exact weighting exponent used in products is unknown (2-4 claimed).", locator: "Detailed description"}
  - {stated_by: reviewer, note: "We do not know that CLEAR-EC labels came from Konan's centre method; the challenge page only names KONAN as an example eye-bank device.", locator: "n/a"}
---

# Method for computing morphology of cornea endothelium cells (US5523213A)

## TL;DR
Konan's 1993/1996 patent describes the "center method": the operator clicks the centre of every cell, and the software estimates each cell's polygon from neighbouring centres. The polygon edges are not midpoint (Voronoi) bisectors: the cut point on each centre-to-neighbour segment is placed by the ratio of the two cells' average neighbour distances raised to a power (2nd to 4th), so cells with larger spacing get disproportionately larger polygons. CD, CV and hexagonality are then computed from these polygons.

## Citation and link
- Takahashi Y, Kasahara T. US 5,523,213 A, Konan Inc., granted 1996-06-04 (priority JP 5-089244, 1993-03-23). Expired.
- Read: https://patents.google.com/patent/US5523213A/en
- Related: Nidek patent US9844316B2 (Sakashita, 2017) describes a "center method" that estimates boundaries from the positional relation of a centre and its peripheral centres and refers to Japanese Patent No. 3265044 for details (https://patents.google.com/patent/US9844316B2/en); it also lets the operator drag a rectangle to restrict automatic detection and prefers >= 100 detected cells.

## Problem and motivation
Corner (vertex) digitising is slow; the centre method needs only one click per cell. The patent aims to estimate realistic polygons from centres so that area, CV and hexagonality can be computed.

## Data
None (patent). The description says the method fitted real cells well when the weighting used the 2nd to 4th power of average distances (no numbers given).

## Method
1. For cell A with centre a, find the nearest centre c; Lmin = |ac|.
2. Candidate neighbours = all centres within 3 x Lmin (claims 2.5-3.5 x), sorted clockwise.
3. Prune: if the angle subtended by two successive neighbours is < ~30 deg, drop the farther one; if the angle across three successive neighbours is > ~128 deg keep the middle one; else drop the middle one when |first-third| < |a-middle|. Repeat until stable.
4. If any gap between successive neighbours exceeds ~100 deg, A is a peripheral ("second-type") cell: excluded from the average-distance calculation and from vertex construction. This is why the centre method excludes the outermost cells of a contiguous group (see [[patel2010flexcenter]]).
5. Average distance d_A = mean distance from a to its retained neighbours.
6. For each neighbour b, place a point on segment ab dividing it in the ratio d_a^k : d_b^k (k between 2 and 4 in the description; claim 6 states a proportional division), draw the perpendicular through it; consecutive perpendiculars intersect at the polygon vertices.
7. Area by polygon formula; CD = 1e6 / mean area; CV = SD / mean area; hexagonal frequency = number of six-vertex polygons / N.

## Results
- No quantitative validation is reported in the patent text (not reported).
- Parameter values and ranges as listed in key_numbers (verified-from-source).

## Limitations
- Author-stated: the centre method needs surrounding peripheral points; cells without enough neighbours get no polygon.
- Reviewer: product software may differ from the patent; the exponent is a range; legibility gaps in the extracted equations.

## Relevance to CD / CV / HEX
- CV: with k = 0 the construction reduces to a Voronoi-like midpoint diagram. Weighting by d^k pushes the shared edge toward the cell with the smaller spacing, which amplifies area differences. A first-order linearisation (our derivation, not in the patent) gives an area-CV amplification of roughly (1 + k) relative to midpoint bisectors when neighbour spacings are uncorrelated; a factor near 2 is inside the plausible range. This is a concrete mechanism for our measured "device CV about twice Voronoi area CV of the same markers".
- HEX: the vertex count of the weighted polygons can differ from the Voronoi six-neighbour count, especially after pruning; HEX matched six-neighbour share within 1-2 % on clean frames, so the effect on HEX may be smaller.
- CD: formula 1e6 / mean area matches our label fact (CD = 1e6 / integer mean area). Peripheral-cell exclusion means the area average covers interior cells only.
- The coordinator's root-cause finding (78-88 % of CV/HEX residual is device outline behaviour plus label noise) makes emulating this algorithm the most direct way to learn device behaviour.

## What it suggests we try
Implement the patent's construction (Lmin, 3xLmin radius, 30/128/100 deg angle rules, d^k cut ratio) as a CPU function of centre points; fit k on the 25 expert-marker frames against the device CV and HEX (k = 0, 1, 2, 3, 4) and see if CV r exceeds 0.89 without the post-hoc 2x scaling. If it does, compute the same device-emulated CV/HEX from detector points (interior cells only, in the most-confident window) and add them as regressor inputs or as a stacked feature.

## Related notes
- [[patel2010flexcenter]] — centre method drops outermost cells; flex-centre includes them and reads CV about 3 points higher.
- [[gasser2015konantopcon]] — Konan CV 37 vs Topcon 29 on the same eyes with manually marked cells: the device algorithm sets the CV scale.
- [[mccarey2008review]] — describes the centre-to-centre method and that ~150 dotted cells yield ~100 areas.
- [[bucht2011simulation]] — a Topcon program showed CV scaling error on simulated images.
- [[vidalvillegas2024eyebank]] — Konan CellChek D in an eye bank: CV/HEX inter-grader ICC <= 0.265.

## Sources
- https://patents.google.com/patent/US5523213A/en
- https://patents.google.com/patent/US9844316B2/en
