---
id: dec-014-close-mask-emulator-self-training
type: decision
name: Close the mask emulator and segmenter self-training for CV and HEX; keep mask CD
summary: 'Coordinator decision 2026-11-09 after the root-cause analysis: our CV and HEX residual variance already equals the expert-centre oracle plus ~150-cell sampling, masks miss more cells than the detector, the emulator adds ~0 on CV and -0.003 to -0.008 on HEX, and the pseudo-label pool holds easy frames. Close both routes for CV and HEX; keep calibrated mask CD as a blend input (CD lane -0.017 on five folds).'
status: accepted
date: '2026-11-09'
deciders:
- coordinator
source: experiments/rootcause_20261109/out/*.json; coordinator message 2026-11-09
decides:
- '[[exp-cd-mask-density-component]]'
alternative:
- '[[idea-segmenter-self-training]]'
- '[[idea-device-emulator-from-masks]]'
motivated_by:
- '[[find-residual-at-oracle-floor-cv-hex]]'
- '[[find-masks-miss-more-than-detector]]'
- '[[find-pseudo-label-pool-easy-frames]]'
- '[[exp-rc-mask-emulator]]'
literature:
- target: '[[reith2025selfadapt]]'
  relation: closes
  note: 'paper marked superseded for CV/HEX; its suggestion (EMA teacher-student with label-based stopping: self-training closed for CV/HEX; CD only if ever)'
- target: '[[pachitariu2025cellposesam]]'
  relation: closes
  note: 'paper marked superseded for CV/HEX; its suggestion (adapt Cellpose-SAM instead of zero-shot masks: closed for CV/HEX (no endothelium data in its training; masks miss 0.260))'
- target: '[[tyagi2025boxcell]]'
  relation: closes
  note: 'paper marked superseded for CV/HEX; its suggestion (prompted SAM pseudo-masks: detector errors propagate; closed for CV/HEX)'
- target: '[[kucharski2021cnnwatershed]]'
  relation: closes
  note: 'paper marked superseded for CV/HEX; its suggestion (CNN markers + watershed areas for CV: segmentation route closed for CV/HEX)'
- target: '[[lee2022mediar]]'
  relation: closes
  note: 'paper marked superseded for CV/HEX; its suggestion (label-gated pseudo-labels for a mask model: self-training closed for CV/HEX)'
- target: '[[qu2020pointnuclei]]'
  relation: closes
  note: 'paper marked superseded for CV/HEX; its suggestion (Voronoi pseudo-labels from points to a border net for area CV: closed for CV/HEX)'
- target: '[[schmidt2018stardist]]'
  relation: closes
  note: 'paper marked superseded for CV/HEX; its suggestion (star-convex polygon head on pseudo-masks for area CV: closed for CV/HEX)'
- target: '[[sierra2023sdm]]'
  relation: closes
  note: 'paper marked superseded for CV/HEX; its suggestion (distance-map head on pseudo-masks for cells: closed for CV/HEX)'
- target: '[[viguerasguillen2022feedbackattn]]'
  relation: closes
  note: 'paper marked superseded for CV/HEX; its suggestion (cell-body head to select cells before area statistics: closed for CV/HEX)'
- target: '[[fabijanska2018unet]]'
  relation: corroborates
  note: segmentation-derived CV carries ~12 % error at 4.5 % count error
- target: '[[ma2024cellsegchallenge]]'
  relation: corroborates
  note: unlabelled images rarely helped segmenters when labelled in-domain data existed
---
# Close the mask emulator and segmenter self-training for CV and HEX; keep mask CD

Coordinator decision 2026-11-09 after the root-cause analysis: our CV and HEX residual variance already equals the expert-centre oracle plus ~150-cell sampling, masks miss more cells than the detector, the emulator adds ~0 on CV and -0.003 to -0.008 on HEX, and the pseudo-label pool holds easy frames. Close both routes for CV and HEX; keep calibrated mask CD as a blend input (CD lane -0.017 on five folds).

## Update 2026-11-09

The stated reason is qualified by [[find-konan-rule-explains-device-cv]]: the oracle floor holds for Voronoi reading; with the Konan outline rule expert centres beat the pipeline on the 25 marker frames. The decision stands: the cell route is bound by centre quality, and masks miss more cells than the detector (0.260 vs 0.150, [[find-masks-miss-more-than-detector]]). Paper notes marked 'superseded for CV/HEX' (nine) link here with relation `closes`.
