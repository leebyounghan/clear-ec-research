---
id: idea-recipe-a-hyperparameters-never-varied
type: idea
name: Recipe A hyperparameters that were never varied
summary: Layer-wise LR decay, weight decay, warm-up length, batch size, joint-loss weights, whole-frame input resolution, more than 20 epochs with the joint loss and appearance augmentation were never varied on recipe A.
status: running
origin: claude
aims_at:
- all
owner_lane: backbone
cost: not estimated in the sources; one recipe A fold takes about 8 h on one GPU
potential_metric: all
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: docs/lane_summaries_20261113/final_summary_backbone_20261113.md section 4 ('Recipe A hyperparameter reads'); docs/FINAL_SUMMARY_20261113.md section 3.3
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[find-recipe-a-changes-converge]]'
- '[[exp-cd-epoch-probe]]'
tested_by:
- target: '[[exp-cd-cdonly-recipe-a-elr1e5]]'
  closeness: related
  note: 'encoder LR 1e-5 with the CD loss only: -0.0030 on folds 0+4 strict, closed'
- target: '[[exp-cd-cdonly-recipe-a-elr3e5]]'
  closeness: related
  note: 'encoder LR 3e-5 with the CD loss only: +0.0181 on folds 0+4 strict, closed; never with the joint loss'
- target: '[[exp-hist-encoder-lr-x5]]'
  closeness: related
  note: 'encoder LR x5 before the lanes: validation 9.11 -> 9.44'
- target: '[[exp-bb-droppath-ema-swa]]'
  closeness: related
  note: 'drop path 0.2 with EMA / SWA: fold 0 +0.027, added -0.002, closed'
- target: '[[exp-bb-vitl16-v2-regularised]]'
  closeness: related
  note: drop path 0.3 + layer-wise decay 0.9 on ViT-L only, stopped at epoch 5; never on recipe A
- target: '[[exp-cd-epoch-probe]]'
  closeness: related
  note: CD single-epoch validation bottoms at epochs 9-12, so a longer schedule was dropped
- target: '[[exp-cd-training-setting-screen-dinov3]]'
  closeness: close
  note: 'running since 22:34 EST 11-13: layer-wise decay 0.75, weight decay 0.05, appearance augmentation 0.5 and 12 epochs + SWA, on DINOv3-B/16 fold 0 (not on recipe A); no value yet'
---

# Recipe A hyperparameters that were never varied

Layer-wise LR decay, weight decay, warm-up length, batch size, joint-loss weights, whole-frame input resolution, more than 20 epochs with the joint loss and appearance augmentation were never varied on recipe A.

## Hypothesis

The backbone lane's final summary lists what was never tried on recipe A (the settings above), next to what was read. What was read: encoder LR only with the CD-only loss (closed), drop path 0.2 (closed), layer-wise decay only on ViT-L (stopped), EMA / SWA (swa6 equals raw inside a 20-epoch run: 9.071 vs 9.068, fold 0), schedule (CD is not falling at the last epoch).

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Minimal test

Not specified in the sources. The runner has `--appear P` for appearance augmentation (P = 0 is pixel-identical to recipe A); it was never run on recipe A.

## Risks

Every recipe A change tested end to end so far landed on the same solution ([[find-recipe-a-changes-converge]]), and a member better alone has not earned stack weight ([[find-better-alone-not-stack-weight]]). Related ideas: [[idea-layerwise-lr-decay]], [[idea-appearance-aug-hex-gap]].

## Outcome 2026-11-13

No run on recipe A itself. After the summaries closed, the CD lane launched a fold-0 screen on DINOv3-B/16 (PI via coordinator; the same runner with all switches off is recipe parity) with three of these settings (layer-wise decay 0.75, weight decay 0.05, appearance augmentation 0.5) and an SWA arm, which is why the status is `running`: [[exp-cd-training-setting-screen-dinov3]]. A separate hyperparameter analysis report (`analysis_hyperparameters_20261113.md`) is not ingested.
