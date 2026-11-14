---
id: find-konan-rule-explains-device-cv
type: finding
name: The device's outline rule (Konan weighted bisectors, k=2) explains device CV ~2x Voronoi; detection quality binds the cell route
summary: On the 25 expert-marker frames the k=2 weighted-bisector rule reads CV r 0.899, LOO-MAPE 7.64 with scale 1.09 (Voronoi 0.887 / 9.29 / 2.08) and HEX r 0.845, MAPE 7.11 (Voronoi 0.783 / 9.12). From detector peaks the same rule reads CV r 0.402 and adds +0.015 to the CV stack. Expert-quality centres would beat the pipeline on these frames, so detection quality is the binding constraint for the cell route.
status: robust
metrics:
- CV
- HEX
date: '2026-11-09'
source: experiments/lane_cv_20261108/RESULTS.tsv lines 85-86 (snapshot 2026-11-09 15:09 UTC); coordinator message 2026-11-09
qualifies:
- target: '[[find-residual-at-oracle-floor-cv-hex]]'
  note: '''better cell reading cannot lower CV/HEX much below the pipeline'' holds for Voronoi reading of centres only: with the device''s own outline rule, expert centres read CV LOO-MAPE 7.64 vs 9.29 (Voronoi) on the 25 frames'
- target: '[[find-device-cv-twice-voronoi]]'
  note: 'the ~2x is the outline rule: scale 2.08 (Voronoi) -> 1.09 (k=2)'
evidence_for:
- target: '[[dec-014-close-mask-emulator-self-training]]'
  note: the cell route is bound by centre quality; masks miss more cells than the detector (0.260 vs 0.150), so closing the mask routes for CV/HEX stands; the decision's stated reason is qualified, not reversed
- target: '[[find-detector-misses-ruin-area-cv]]'
  note: detector misses swamp the k=2 gain (r 0.402 vs 0.899 on expert centres)
motivated_by:
- '[[find-device-cv-twice-voronoi]]'
- '[[find-expert-markers-read-cv]]'
literature:
- target: '[[takahashi1996konanpatent]]'
  relation: corroborates
  note: the patent's weighted bisectors (k 2-4) reproduce the device CV scale; k=2 fitted
- target: '[[gasser2015konantopcon]]'
  relation: corroborates
  note: the CV scale is set by the device's polygon construction (Konan 37 vs Topcon 29 on the same eyes)
---
# The device's outline rule (Konan weighted bisectors, k=2) explains device CV ~2x Voronoi; detection quality binds the cell route

On the 25 expert-marker frames the k=2 weighted-bisector rule reads CV r 0.899, LOO-MAPE 7.64 with scale 1.09 (Voronoi 0.887 / 9.29 / 2.08) and HEX r 0.845, MAPE 7.11 (Voronoi 0.783 / 9.12). From detector peaks the same rule reads CV r 0.402 and adds +0.015 to the CV stack. Expert-quality centres would beat the pipeline on these frames, so detection quality is the binding constraint for the cell route.

## Evidence

- [[exp-cv-konan-bisectors-markers]]: 25 expert-marker frames, k chosen leave-one-out (k=2 on 25/25): CV r 0.899, LOO-MAPE 7.64, device/estimate scale 1.09; HEX r 0.845, MAPE 7.11; Voronoi k=0: CV 0.887 / 9.29 / 2.08, HEX 0.783 / 9.12.
- [[exp-cv-konan-bisectors-detector]]: 3-seed TTA detector peaks, all frames: CV r 0.402 with the label (Voronoi 0.407), r 0.028 with the stack residual; as a nested correction on the one-recipe stack (CV 10.114) +0.015 (worse). hex_k2 r 0.472 with the HEX label, -0.008 with the best-pipeline HEX residual.
- On the same 25 frames hybrid B reads CV r 0.53 ([[find-expert-markers-read-cv]]), so expert centres with the device rule beat the pipeline there (coordinator).

## Limits

25 frames of one style; k picked leave-one-out on the same frames. The detector read is one detector set (3-seed TTA peaks, threshold 0.35). Says nothing about masks beyond what [[find-masks-miss-more-than-detector]] already says. The root-cause finding is qualified, not replaced: its variance budget used Voronoi on expert centres.

## Bears on

Closes the detector route of [[idea-device-outline-rule]] and [[idea-interior-outer-ring-features]] (reopen only if detection reaches near-expert quality). Raises the value of anything that improves centre quality itself ([[idea-peak-repair-from-image]]), within what [[exp-cv-spacing-prior-nms]] and [[exp-hist-graph-repair]] already closed.
