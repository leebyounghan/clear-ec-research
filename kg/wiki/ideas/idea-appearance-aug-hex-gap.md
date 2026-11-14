---
id: idea-appearance-aug-hex-gap
type: idea
name: Appearance augmentation against the HEX test gap, without batch-norm re-estimation
summary: Train with stacked appearance augmentation (gamma, brightness/contrast, radial vignetting, blur/sharpen, noise; no scaling) so the regressors are robust to acquisition shift, aimed at submission 1's +0.40 HEX test-minus-OOF gap; avoid test-time BN statistics, which hurt regression in published TTA work.
status: dropped
origin: literature
aims_at:
- HEX
- CV
owner_lane: hex
cost: ~0.5 CPU-h to read the CV lane's queued DINOv3 aug-bundle member in the HEX chain; ~10 GPU-h for a dedicated two-fold version
cost_hours: 0.5
cost_unit: CPU-h
potential_metric: HEX
potential_gain: 0.02
potential_basis: 'speculative estimate: 5 % of submission 1''s +0.395 HEX test-minus-OOF gap; OOF cannot measure it (the per-source OOF split is the only proxy), and the old bundle read HEX +0.03 on one fold'
source: Literature review 2026-11-09 (paper notes' suggests_try), coordinator list of eight suggestions, suggestion 8
novelty_check: 'KG check 2026-11-09 (kg_query check + grep over wiki/): check augmentation; check bundle; check test gap -> exp-hist-augmentation-bundle (closed on the old gate), exp-gpu0-measured-augmentation (closed), exp-cv-dinov3-aug-bundle (queued 11-09), idea-test-time-adaptation-regression (never tried). Literature: papers from the KG literature review (zhang2020bigaug, adachi2025ssa, tellez2019augnorm, ma2024cellsegchallenge, wang2021tent); no new web search in this pass.'
motivated_by:
- '[[find-hex-test-gap]]'
- '[[find-recent-images-darker]]'
- '[[find-oof-test-trust]]'
builds_on:
- '[[method-measured-augmentation]]'
tested_by:
- target: '[[exp-bb-shading-correction]]'
  closeness: related
  note: closed together with shading correction; the appearance-augmentation arm itself was never run
- target: '[[exp-cv-hex-test-gap-composition]]'
  closeness: related
  note: the HEX test gap it aims at is inside 1000-image sampling noise
- target: '[[exp-cv-dinov3-aug-bundle]]'
  closeness: close
  note: 'closed for CV: weight 0, inside the seed mean -0.030 vs -0.037; no HEX read of it is recorded'
- target: '[[exp-hist-augmentation-bundle]]'
  closeness: close
  note: 'recipe A fold 0, seed 46: mean -0.042, CD -0.08, CV -0.07, HEX +0.03; closed only on the old CI gate'
- target: '[[exp-gpu0-measured-augmentation]]'
  closeness: close
  note: 'augmentation within the measured range of real frames: +0.004, closed'
literature:
- target: '[[zhang2020bigaug]]'
  relation: source-of-method
  note: stacked appearance transforms; brightness and contrast are the strongest single transforms
- target: '[[adachi2025ssa]]'
  relation: explains
  note: BN-statistics adaptation can sharply reduce regression R2
- target: '[[tellez2019augnorm]]'
  relation: suggests
  note: appearance augmentation is the must-have; normalisation optional
- target: '[[ma2024cellsegchallenge]]'
  relation: suggests
  note: diverse intensity augmentation helped the cell-segmentation challenge winners
- target: '[[wang2021tent]]'
  relation: explains
  note: test-time BN updates fail under some shifts (MNIST->SVHN)
---
# Appearance augmentation against the HEX test gap, without batch-norm re-estimation

## Hypothesis

Part of the HEX test gap is acquisition shift (darker, vignetted frames); training-time appearance augmentation narrows it at no inference cost, while test-time BN re-estimation is risky for regression heads.

## Closest tries in the KG

- [[exp-cv-dinov3-aug-bundle]] (close, running in the CV lane).
- [[exp-hist-augmentation-bundle]] (close, closed on the old gate; HEX +0.03 on one fold).
- [[exp-gpu0-measured-augmentation]] (close, closed).

## Literature

zhang2020bigaug, adachi2025ssa, tellez2019augnorm, ma2024cellsegchallenge, wang2021tent.

## Minimal test

Read the CV lane's aug-bundle member (five folds when done) in the HEX chain at share 0.2 and on the old-vs-new source split; if it lowers HEX or the per-source gap, a dedicated HEX member with a vignetting field follows. The encoders use LayerNorm, so BN re-estimation is not in play for them; keep any CNN detector in eval mode.

## Risks

OOF cannot show a test-gap gain; augmentation can cost OOF HEX (+0.03 in the old read). Per-source evaluation is a diagnostic only, never a per-source choice ([[con-no-source-id-shortcuts]]).

## Outcome 2026-11-11

Dropped without a run (backbone lane notes 2026-11-11 05:15 EST: 'closed with shading, never run'). The gap it aims at is inside sampling noise ([[find-test-gap-within-sampling-noise]]) and the source shift is a level offset ([[find-source-shift-level-offset]]).

## Update 2026-11-13

Not run against the HEX gap. The backbone lane's final summary lists appearance augmentation as 'prepared, never run' (`--appear P` exists in the runner; P = 0 is pixel-identical to recipe A); it is one item of [[idea-recipe-a-hyperparameters-never-varied]]. The CV lane's augmentation-bundle member is closed for CV. After the summaries closed, an `--appear 0.5` arm on DINOv3-B/16 fold 0 was launched for CD (22:34 EST 11-13) and had no value at the last ledger line ([[exp-cd-training-setting-screen-dinov3]]); its read is CD, not the HEX test gap.
