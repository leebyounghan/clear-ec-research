---
id: idea-device-outline-rule
type: idea
name: 'Device outline rule from points: Konan weighted bisectors, exponent fitted on the 25 marker frames'
summary: 'Fit the exponent of the Konan centre-method weighted bisectors on the 25 expert-marker frames, then compute device-style CV and HEX from detector points as features. Closed for the detector route: expert centres read CV LOO-MAPE 7.64 (scale 1.09) but detector peaks add +0.015 to the CV stack.'
status: closed
origin: literature
aims_at:
- CV
- HEX
owner_lane: cv
cost: ~1.5 CPU-h as run (25-frame fit + detector-peak features + nested stack read)
cost_hours: 1.5
cost_unit: CPU-h
potential_metric: CV
potential_gain: null
potential_basis: 'closed: the detector route read +0.015 (worse) on the CV stack; reopen only if detection reaches near-expert quality'
source: Literature review 2026-11-09 (paper notes' suggests_try), coordinator list of eight suggestions, suggestion 1
novelty_check: 'KG check 2026-11-09 (kg_query check + grep over wiki/): check konan; check voronoi cv; check emulator -> exp-rc-mask-emulator (masks, closed by dec-014), exp-hist-consensus-voronoi-cv and exp-hist-simple-cv-estimators (closed, numbers not recorded), idea-device-formula-audit (open); no weighted-bisector try before the CV lane ran lines 85-86. Literature: papers from the KG literature review (takahashi1996konanpatent, gasser2015konantopcon, mccarey2008review, patel2010flexcenter, bucht2011simulation); no new web search in this pass.'
motivated_by:
- '[[find-device-cv-twice-voronoi]]'
- '[[find-expert-markers-read-cv]]'
- '[[find-residual-at-oracle-floor-cv-hex]]'
builds_on:
- '[[method-konan-weighted-bisectors]]'
- '[[ds-expert-markers-25]]'
- '[[method-pre-only-detector]]'
tested_by:
- target: '[[exp-hex-cellroute-konan-step]]'
  closeness: close
  note: 'added 2026-11-11: Konan statistics from a union point set as a nested HEX correction, kept in the freeze (D-HEX-005): 9.70581 as computed vs 9.71882; end to end strict 9.71342 vs 9.71977. The CV route stays closed'
- target: '[[exp-cv-konan-bisectors-markers]]'
  closeness: same
  note: 'the 25-frame fit: k=2, CV r 0.899, LOO-MAPE 7.64, scale 1.09; HEX r 0.845, MAPE 7.11 (gate passed)'
- target: '[[exp-cv-konan-bisectors-detector]]'
  closeness: same
  note: 'the detector route: k=2 CV from 3-seed TTA peaks, r 0.402, correction +0.015 on the CV stack; closed for CV'
- target: '[[exp-rc-mask-emulator]]'
  closeness: related
  note: device-style CV/HEX from masks (ok/interior/Voronoi cells), closed by dec-014
literature:
- target: '[[takahashi1996konanpatent]]'
  relation: source-of-method
  note: weighted bisectors, k 2-4
- target: '[[gasser2015konantopcon]]'
  relation: suggests
  note: device construction sets the CV scale
- target: '[[mccarey2008review]]'
  relation: source-of-method
  note: 'centre method: interior cells only, dot jitter inflates area SD'
- target: '[[clearec2026challenge]]'
  relation: suggests
  note: organisers name KONAN as the eye-bank device; no label algorithm published
---
# Device outline rule from points: Konan weighted bisectors, exponent fitted on the 25 marker frames

## Hypothesis

The device computes CV and HEX from centre points with weighted bisectors, not Voronoi; emulating it removes the ~2x CV scale and, from detector points, could give a device-style feature the regressor lacks.

## Closest tries in the KG

- [[exp-cv-konan-bisectors-markers]] (same, measurement): expert centres, k=2 picked LOO 25/25: CV r 0.899, LOO-MAPE 7.64, scale 1.09; HEX r 0.845, MAPE 7.11 vs Voronoi 0.887 / 9.29 / 2.08 and 0.783 / 9.12.
- [[exp-cv-konan-bisectors-detector]] (same, closed): from detector peaks CV r 0.402 (Voronoi 0.407), r 0.028 with the stack residual, +0.015 on the CV stack (10.114, five folds).
- [[exp-rc-mask-emulator]] (related, closed): device-style values from masks; CV ~0, HEX -0.003 to -0.008.

## Literature

takahashi1996konanpatent (construction), gasser2015konantopcon (device sets the CV scale), mccarey2008review (interior cells, ~150 dotted centres), patel2010flexcenter (centre vs flex-centre), bucht2011simulation (Topcon CV scaling error).

## Minimal test

Done (CV lane lines 85-86). HEX: per-frame hex_k2 values are with the HEX lane (out/konan_feats_t0.35.npz; r -0.008 with the best-pipeline HEX residual); a chain read costs CPU minutes.

## Risks

Status `closed` for the detector route: reopen only if detection reaches near-expert quality ([[find-konan-rule-explains-device-cv]]).

## Outcome 2026-11-11

The HEX use of the rule is kept as the cell-route step ([[exp-hex-cellroute-konan-step]]); the CV route stays closed ([[exp-hex-cellroute-closed-variants]]).
