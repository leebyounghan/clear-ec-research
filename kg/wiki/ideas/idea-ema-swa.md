---
id: idea-ema-swa
type: idea
name: EMA or SWA of weights
summary: Weight averaging captures the -0.045 seen from averaging stored predictions.
status: partially-tested
origin: claude
aims_at:
- all
owner_lane: backbone
cost: ~0 extra (applied inside runs already planned)
cost_hours: 0.5
cost_unit: GPU-h
potential_metric: all
potential_gain: 0.001
potential_basis: 'estimate: SWA gained nothing over raw inside 12-epoch runs; the -0.045 prediction-averaging effect is already captured by snapshot/TTA averaging (CD -0.004)'
source: 'docs/LANES_20261108.md ''Facts per lane: Backbone'''
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-checkpoint-averaging-expectation]]'
builds_on:
- '[[method-checkpoint-averaging]]'
tested_by:
- target: '[[exp-bb-droppath-ema-swa]]'
  closeness: close
  note: with drop path; SWA gains nothing over raw inside a 12-epoch run
- target: '[[exp-bb-checkpoint-average]]'
  closeness: related
  note: prediction averaging on stored epochs, -0.045
- target: '[[exp-cd-flip-tta-snapshot]]'
  closeness: related
  note: epoch 8-19 snapshot kept on CD (-0.004)
- target: '[[exp-cd-deploy-tiers]]'
  closeness: related
  note: CD t1 drops the epoch 8-19 snapshot at no strict cost (6.1737 vs 6.1743), so an SWA average in its place is moot
- target: '[[exp-cd-vitl-averaged-tracks-in-stack]]'
  closeness: same
  note: 'ViT-L slot from the swa6 / ema999 tracks instead of raw, strict five folds: +0.0066 / +0.0028 against 6.1737; better alone, worse in the stack'
- target: '[[exp-cd-training-setting-screen-dinov3]]'
  closeness: close
  note: 'arm D, 12 epochs + SWA from epoch 6 on DINOv3-B/16 fold 0: running, no value yet'
---
# EMA or SWA of weights

## Hypothesis

Weight averaging captures the -0.045 seen from averaging stored predictions.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Update 2026-11-13

From the backbone lane's final summary: inside a 20-epoch recipe A run swa6 equals raw (9.071 vs 9.068, fold 0); on ViT-L swa6 beats raw (two folds 9.115 vs 9.231; synthetic ViT-L 9.0495 vs 9.1654). The CD lane lane's summary listed 'SWA average instead of epoch snapshots' as not run and moot after tier t1 ([[find-cd-full-extra-weights-nothing-over-t1]]); an hour later the averaged ViT-L tracks were read in the stack at the PI's request: swa6 +0.0066 (CI +0.001 to +0.013), ema999 +0.0028 (CI -0.002 to +0.007), strict five folds against 6.1737 ([[exp-cd-vitl-averaged-tracks-in-stack]]). An SWA arm on DINOv3-B/16 fold 0 was running at the last ledger line. Status stays partially-tested.
