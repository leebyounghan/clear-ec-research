---
id: idea-localised-pooling-added
type: idea
name: Add a legible-window view to the whole frame instead of replacing it
summary: Localised pooling was closed when it replaced the whole frame; as an added member it was never read.
status: closed
origin: claude
aims_at:
- CV
- HEX
owner_lane: backbone
cost: 'spent: ViT-L window on folds 0,4 and recipe A window on five folds'
cost_hours: 4
cost_unit: GPU-h
potential_metric: CV
potential_gain: 0.007
potential_basis: 'estimate: the ViT-L window added CV -0.018 vs the best pipeline on folds 0,4 but only -0.007 on top of whole-frame ViT-L (-0.021 -> -0.028); two-fold reads overstate'
source: docs/ASSUMPTION_AUDIT_20261106.md belief 9
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-technician-region]]'
builds_on:
- '[[method-legible-window-crop]]'
tested_by:
- target: '[[exp-hist-localised-pooling]]'
  closeness: close
  note: replaced the whole frame
- target: '[[exp-cv-legible-window-crop]]'
  closeness: close
  note: crop member lost cells, +0.03
- target: '[[exp-bb-vitl16-window-view]]'
  closeness: same
  note: 'ViT-L on the 2x window, folds 0,4: added -0.0085 (CV -0.018); on top of whole-frame ViT-L CV only, CD/HEX worse'
- target: '[[exp-bb-recipe-a-window-view]]'
  closeness: same
  note: 'third window version, recipe A on the 2x window, five folds: CD stack +0.0026 (6.1751 vs 6.1726), HEX chain +0.0002; closed'
literature:
- target: '[[joseph2020postkeratoplasty]]'
  relation: suggests
  note: 'the label is a sub-region statistic: restrict features to the box'
---
# Add a legible-window view to the whole frame instead of replacing it

## Hypothesis

Localised pooling was closed when it replaced the whole frame; as an added member it was never read.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-13

Closed: as an added member the window view does not help. ViT-L on the window adds CV only on top of the whole-frame ViT-L (two folds), and recipe A on the window reads CD +0.0026 on five folds and HEX +0.0002 in the chain. Native tiles from the legible-window region are closed for CD as well ([[exp-bb-native-tile-cd-only]]).
