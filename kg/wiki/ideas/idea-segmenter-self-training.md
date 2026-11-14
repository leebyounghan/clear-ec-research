---
id: idea-segmenter-self-training
type: idea
name: Adapt the segmenter on our frames with label-filtered pseudo-labels
summary: Teacher-student adaptation of Cellpose-SAM on frames whose mask CD/HEX agree with the labels, to read cells better for CV and HEX.
status: closed
origin: literature
aims_at:
- CV
- HEX
source: paper notes reith2025selfadapt, tyagi2025boxcell, pachitariu2025cellposesam; root-cause analysis 2026-11-09
novelty_check: 'Literature: SelfAdapt (reith2025selfadapt) is the closest recipe. Closed 2026-11-09 by dec-014 before running a full adaptation.'
motivated_by:
- '[[find-masks-miss-more-than-detector]]'
builds_on:
- '[[method-segmenter-self-training]]'
- '[[method-cellpose-sam-masks]]'
tested_by:
- target: '[[exp-rc-pseudo-label-pool]]'
  closeness: close
  note: the pool exists (~890 frames per fold) but holds easy frames
- target: '[[exp-rc-variance-budget]]'
  closeness: related
  note: CV/HEX residual already at the oracle-plus-sampling floor
literature:
- target: '[[reith2025selfadapt]]'
  relation: source-of-method
- target: '[[tyagi2025boxcell]]'
  relation: suggests
- target: '[[pachitariu2025cellposesam]]'
  relation: source-of-method
- target: '[[qu2020pointnuclei]]'
  relation: suggests
  note: segmentation from partial centre points with Voronoi labels
- target: '[[schmidt2018stardist]]'
  relation: suggests
  note: polygon head on filtered pseudo-masks; closed for CV/HEX by dec-014
- target: '[[lee2022mediar]]'
  relation: suggests
  note: gate pseudo-labels by image labels + replay; closed for CV/HEX by dec-014
- target: '[[sierra2023sdm]]'
  relation: suggests
  note: distance-map head on pseudo-masks; closed for CV/HEX by dec-014
- target: '[[kervadec2019constrained]]'
  relation: suggests
  note: CD label as a size constraint on soft masks; CV/HEX route closed by dec-014
---
# Adapt the segmenter on our frames with label-filtered pseudo-labels

## Hypothesis

Teacher-student adaptation of Cellpose-SAM on frames whose mask CD/HEX agree with the labels, to read cells better for CV and HEX.

## Notes

Closed for CV and HEX only. Calibrated mask CD stays in the CD blend.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
