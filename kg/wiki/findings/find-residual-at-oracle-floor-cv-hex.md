---
id: find-residual-at-oracle-floor-cv-hex
type: finding
name: Our CV and HEX residual variance already equals expert-centre error plus ~150-cell sampling
summary: 'Five-fold OOF residual log variance vs an oracle (expert-marker Voronoi in the true region, 25 frames) plus sampling: CV 0.0190 vs 0.0189; HEX 0.0154 vs 0.0191. Sampling is 22 % (CV) and 36 % (HEX) of our residual variance; what expert centres in the true region still miss (device behaviour plus label noise) is 78 % and 88 %.'
status: tentative
metrics:
- CV
- HEX
date: '2026-11-09'
source: experiments/rootcause_20261109/out/q3_decomp.json, out/q3b_oracle_logsd.json; coordinator summary 2026-11-09
qualifies:
- '[[find-hex-near-ceiling]]'
literature:
- target: '[[viguerasguillen2019regression]]'
  relation: corroborates
  note: HEX hardest for direct regression
- target: '[[tofallis2015logratio]]'
  relation: explains
  note: log-ratio residuals as the variance scale
- target: '[[kucharski2021cnnwatershed]]'
  relation: closes
  note: 'better segmentation cannot lower CV/HEX much below the pipeline (Voronoi reading of centres); qualified by find-konan-rule-explains-device-cv: centre quality binds'
- target: '[[lee2022mediar]]'
  relation: closes
  note: 'better segmentation cannot lower CV/HEX much below the pipeline (Voronoi reading of centres); qualified by find-konan-rule-explains-device-cv: centre quality binds'
- target: '[[pachitariu2025cellposesam]]'
  relation: closes
  note: 'better segmentation cannot lower CV/HEX much below the pipeline (Voronoi reading of centres); qualified by find-konan-rule-explains-device-cv: centre quality binds'
- target: '[[qu2020pointnuclei]]'
  relation: closes
  note: 'better segmentation cannot lower CV/HEX much below the pipeline (Voronoi reading of centres); qualified by find-konan-rule-explains-device-cv: centre quality binds'
- target: '[[reith2025selfadapt]]'
  relation: closes
  note: 'better segmentation cannot lower CV/HEX much below the pipeline (Voronoi reading of centres); qualified by find-konan-rule-explains-device-cv: centre quality binds'
- target: '[[schmidt2018stardist]]'
  relation: closes
  note: 'better segmentation cannot lower CV/HEX much below the pipeline (Voronoi reading of centres); qualified by find-konan-rule-explains-device-cv: centre quality binds'
- target: '[[sierra2023sdm]]'
  relation: closes
  note: 'better segmentation cannot lower CV/HEX much below the pipeline (Voronoi reading of centres); qualified by find-konan-rule-explains-device-cv: centre quality binds'
- target: '[[tyagi2025boxcell]]'
  relation: closes
  note: 'better segmentation cannot lower CV/HEX much below the pipeline (Voronoi reading of centres); qualified by find-konan-rule-explains-device-cv: centre quality binds'
- target: '[[viguerasguillen2022feedbackattn]]'
  relation: closes
  note: 'better segmentation cannot lower CV/HEX much below the pipeline (Voronoi reading of centres); qualified by find-konan-rule-explains-device-cv: centre quality binds'
- target: '[[desanctis2006twodevices]]'
  relation: corroborates
  note: 'inter-examiner ICC < 0.65 for CV/HEX: reader and device behaviour dominate'
- target: '[[vidalvillegas2024eyebank]]'
  relation: corroborates
  note: eye-bank inter-grader ICC <= 0.265 for CV/HEX
- target: '[[ishida2023bayeserror]]'
  relation: explains
  note: estimate the floor from the label-generating noise; a floor from an uncertain noise model is itself biased
- target: '[[scarpa2016morphometry]]'
  relation: explains
  note: inter-expert floor on the same cells (CV 2.9 %, HEX 2.6 %)
---
# Our CV and HEX residual variance already equals expert-centre error plus ~150-cell sampling

Five-fold OOF residual log variance vs an oracle (expert-marker Voronoi in the true region, 25 frames) plus sampling: CV 0.0190 vs 0.0189; HEX 0.0154 vs 0.0191. Sampling is 22 % (CV) and 36 % (HEX) of our residual variance; what expert centres in the true region still miss (device behaviour plus label noise) is 78 % and 88 %.

## Limits and notes

Against the 20261106 best pipeline. The oracle part rests on 25 frames of one style (2023); the sampling term is BRIEF's independent-draw SE; shares are of our residual variance (HEX sums to 124 %). Better cell reading alone cannot lower CV/HEX much below the current pipeline; the remaining error is mostly device behaviour and label noise.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.

## Qualified 2026-11-09

[[find-konan-rule-explains-device-cv]] qualifies this finding (it does not replace it): the oracle here reads expert centres with Voronoi. With the device's own outline rule (Konan weighted bisectors, k=2) expert centres read CV LOO-MAPE 7.64 vs 9.29 for Voronoi on the 25 frames, so expert-quality centres would beat the pipeline there. 'Cell reading is not the limit' holds for Voronoi reading only; for the cell route the binding constraint is detection quality (detector peaks: CV r 0.402, +0.015 on the CV stack).
