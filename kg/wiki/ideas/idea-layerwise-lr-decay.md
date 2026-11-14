---
id: idea-layerwise-lr-decay
type: idea
name: Layer-wise LR decay under recipe A
summary: LLRD keeps pretrained features while the head adapts.
status: partially-tested
origin: claude
aims_at:
- all
owner_lane: backbone
cost: ~12 GPU-h
cost_hours: 12
cost_unit: GPU-h
potential_metric: all
potential_gain: 0.001
potential_basis: 'estimate: the only read (ViT-L v2 with drop path 0.3 + LLRD 0.9) was worse at epoch 5 and stopped; recipe A changes converge to the same solution (r 0.98-0.99)'
source: 'docs/LANES_20261108.md ''Facts per lane: Backbone'''
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[exp-hist-encoder-lr-x5]]'
builds_on:
- '[[method-recipe-a]]'
- '[[method-drop-path-regularisation]]'
tested_by:
- target: '[[exp-bb-vitl16-v2-regularised]]'
  closeness: close
  note: ViT-L with drop path 0.3 + LLRD 0.9, stopped at epoch 5 (10.40 vs ~9.3); never on recipe A
- target: '[[exp-cd-training-setting-screen-dinov3]]'
  closeness: close
  note: 'arm A, layer-wise decay 0.75 (top block 1e-5, head 1e-3) on DINOv3-B/16 fold 0: running since 22:34 EST 11-13, no value yet'
---
# Layer-wise LR decay under recipe A

## Hypothesis

LLRD keeps pretrained features while the head adapts.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Update 2026-11-13

Still never tried on recipe A itself; it is one item of [[idea-recipe-a-hyperparameters-never-varied]]. A layer-wise decay arm (0.75) on DINOv3-B/16 fold 0 was launched at 22:34 EST 11-13 and had no value at the last ledger line ([[exp-cd-training-setting-screen-dinov3]]).
