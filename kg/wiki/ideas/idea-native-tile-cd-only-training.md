---
id: idea-native-tile-cd-only-training
type: idea
name: Native-resolution random-tile training with tile-averaged inference for CD
summary: Many native-resolution tiles per frame instead of one whole-frame sample, CD loss only, averaged over a tile grid at inference.
status: closed
origin: claude
aims_at:
- CD
owner_lane: backbone
cost: about 1.5 GPU-h for folds 0 and 4
cost_hours: 1.5
cost_unit: GPU-h
potential_metric: CD
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: shared notes ledger_20261111.jsonl D-CDSIMPLE-001 (line 91), A-BB-TILE-F04 (line 114), O-CD-TILE-001 (line 117)
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[exp-ref-submission1]]'
- '[[find-technician-region]]'
tested_by:
- target: '[[exp-bb-native-tile-cd-only]]'
  closeness: same
  note: member CD 6.944 vs plain DINOv3-B 6.689 on folds 0+4 (+0.255)
- target: '[[exp-cd-native-tile-stack-read]]'
  closeness: same
  note: added +0.0001, in place of the DINOv3 seed mean +0.0054, averaged -0.0018 (folds 0+4 strict)
- target: '[[exp-cv-legible-window-crop]]'
  closeness: related
  note: a crop member at the legible window was worse for CV as well
uses_method:
- '[[method-native-tile-training]]'
---

# Native-resolution random-tile training with tile-averaged inference for CD

Many native-resolution tiles per frame instead of one whole-frame sample, CD loss only, averaged over a tile grid at inference.

## Hypothesis

One of the simple CD approaches the KG showed as never tested (D-CDSIMPLE-001 b).

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-13

Closed for CD (D-CDTRAIN-003).
