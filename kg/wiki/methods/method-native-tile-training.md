---
id: method-native-tile-training
type: method
name: Native-resolution random-tile training with tile-averaged inference
summary: DINOv3-B trained on 4 random native 448 px crops per frame per epoch from the legible-window region with a log-CD loss only; inference averages log CD over a 3 x 4 overlapping tile grid (legible tiles).
code: lane_backbone_20261108/source/bb_runner_tile_20261111.py
variant_of:
- '[[method-legible-window-crop]]'
---

# Native-resolution random-tile training with tile-averaged inference

DINOv3-B trained on 4 random native 448 px crops per frame per epoch from the legible-window region with a log-CD loss only; inference averages log CD over a 3 x 4 overlapping tile grid (legible tiles).

## Where it is used

Experiments link here with `uses_method`; see backlinks. Sources: ledger D-CDSIMPLE-001 (b), O-BB-TILE-PLAN; `lane_backbone_20261108/source/bb_runner_tile_20261111.py`.
