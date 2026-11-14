---
id: method-konan-weighted-bisectors
type: method
name: Konan centre-method weighted bisectors
aliases:
- Konan centre method
- weighted-bisector polygons
- device outline rule
summary: Cell polygons from cell centres with power-weighted bisectors (cut point at the ratio of the two cells' mean neighbour distances to the power k; k=0 is Voronoi), neighbours within ~3x the nearest-centre distance, angle pruning, peripheral cells excluded; CV and HEX from the polygons.
code: lane_cv_20261108 (konan features out/konan_feats_t0.35.npz; script named in the CV lane folder)
variant_of:
- '[[method-expert-marker-oracle]]'
part_of: []
builds_on:
- '[[takahashi1996konanpatent]]'
uses_dataset:
- '[[ds-expert-markers-25]]'
literature:
- target: '[[takahashi1996konanpatent]]'
  relation: source-of-method
  note: US5523213A; exponent 2-4 in the patent text, k=2 fitted on our 25 marker frames
- target: '[[patel2010flexcenter]]'
  relation: explains
  note: centre method drops the outermost cells; flex-centre includes them
- target: '[[mccarey2008review]]'
  relation: explains
  note: ~150 dotted centres give ~100 interior areas
---
# Konan centre-method weighted bisectors

## What it is

The construction described in the Konan patent (US5523213A): for each centre, neighbours within ~3x the nearest-centre distance, pruned by angle rules; cells with a large angular gap are peripheral and get no polygon; each edge is the perpendicular through a point dividing the centre-to-neighbour segment in the ratio d_a^k : d_b^k of the two cells' mean neighbour distances. k=0 reduces to midpoint (Voronoi-like) bisectors.

## Where it is used

Experiments link here with `uses_method`: [[exp-cv-konan-bisectors-markers]] (25 expert-marker frames), [[exp-cv-konan-bisectors-detector]] (detector peaks, all frames).

## Known numbers

On the 25 expert-marker frames, k=2 (LOO pick 25/25): CV r 0.899, LOO-MAPE 7.64, scale 1.09; HEX r 0.845, MAPE 7.11 (Voronoi 0.887 / 9.29 / 2.08 and 0.783 / 9.12). From detector peaks: CV r 0.402 (Voronoi 0.407), correction +0.015 on the CV stack.
