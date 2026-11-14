---
id: idea-interior-outer-ring-features
type: idea
name: Interior cells against interior plus outer ring (centre vs flex-centre) as two feature variants
summary: 'Compute device-style CV/HEX twice, from interior cells only (centre method) and with the outermost ring included (flex-centre), and use the better or both as features. Closed: the two variants were read from masks (closed for CV/HEX by dec-014), and the detector-point route of the same rule adds nothing.'
status: closed
origin: literature
aims_at:
- CV
- HEX
owner_lane: cv
cost: < 0.5 CPU-h on top of the Konan features
cost_hours: 0.5
cost_unit: CPU-h
potential_metric: CV
potential_gain: null
potential_basis: 'closed: mask variants CV ~0 (exp-rc-mask-emulator); Konan polygons from detector peaks (interior cells by construction) +0.015'
source: Literature review 2026-11-09 (paper notes' suggests_try), coordinator list of eight suggestions, suggestion 4
novelty_check: 'KG check 2026-11-09 (kg_query check + grep over wiki/): check interior; check flex -> exp-rc-mask-emulator (ok / interior / Voronoi cells from masks), exp-cv-konan-bisectors-detector (peripheral cells excluded by construction). Literature: papers from the KG literature review (patel2010flexcenter, mccarey2008review, vidalvillegas2024eyebank); no new web search in this pass.'
motivated_by:
- '[[find-device-cv-twice-voronoi]]'
- '[[idea-device-outline-rule]]'
depends_on:
- '[[idea-device-outline-rule]]'
builds_on:
- '[[method-konan-weighted-bisectors]]'
- '[[method-mask-emulator]]'
tested_by:
- target: '[[exp-rc-mask-emulator]]'
  closeness: same
  note: 'interior vs ok (all valid) cells from masks, calibrated and added: CV -0.0026 to +0.0064, HEX -0.0028 to -0.0076; closed for CV/HEX by dec-014'
- target: '[[exp-cv-konan-bisectors-detector]]'
  closeness: close
  note: 'interior-cell (centre-method) polygons from detector peaks: +0.015; the flex-centre variant from points was not run'
- target: '[[exp-cv-konan-bisectors-markers]]'
  closeness: related
  note: on expert centres the centre-method rule matches the label scale (1.09)
literature:
- target: '[[patel2010flexcenter]]'
  relation: source-of-method
  note: flex-centre CV 33 vs centre 30 on the same normal eyes
- target: '[[mccarey2008review]]'
  relation: explains
  note: centre method yields ~100 interior areas from ~150 dots
---
# Interior cells against interior plus outer ring (centre vs flex-centre) as two feature variants

## Hypothesis

The device mode (which cells enter the polygon statistics) shifts CV by ~10 % relative; matching it should sharpen device-style features.

## Closest tries in the KG

- [[exp-rc-mask-emulator]] (same variants on masks, closed).
- [[exp-cv-konan-bisectors-detector]] (close: interior-only polygons from detector peaks, +0.015).

## Literature

patel2010flexcenter, mccarey2008review.

## Minimal test

None while the detector route is closed. Reopen with idea-device-outline-rule if detection reaches near-expert quality: on the 25 marker frames compare centre vs flex-centre CV against the label first.

## Risks

Detector misses dominate either variant ([[find-detector-misses-ruin-area-cv]]).
