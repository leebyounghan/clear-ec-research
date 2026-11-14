---
id: idea-vigueras-guillen-weights
type: idea
name: Use the public Vigueras-Guillen weights (on the pod, untested)
summary: Weights trained for the Rotterdam papers as a detector or segmenter (GPL-3.0, about 1.04 um/px vs 0.7716).
status: closed
origin: claude
aims_at:
- CV
- HEX
owner_lane: cv
cost: ~2 GPU-h (detector/segmenter inference + read)
cost_hours: 2
cost_unit: GPU-h
potential_metric: CV
potential_gain: 0.002
potential_basis: 'estimate: a segmenter route for CV/HEX is closed by dec-014; as a detector it would need to beat the pre-only detector''s miss 0.150 to matter (find-konan-rule-explains-device-cv)'
source: memory/clear-ec-external-data-survey-20261106.md
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-masks-miss-more-than-detector]]'
builds_on:
- '[[viguerasguillen2020dsaek]]'
- '[[viguerasguillen2019fcn]]'
tested_by:
- target: '[[exp-cv-vigueras-seg-features]]'
  closeness: close
  note: 'segmentation features on the CV stack: +0.0196'
- target: '[[exp-hex-vigueras-edge-network]]'
  closeness: close
  note: 'edge / ROI network features in the HEX chain: +0.0003, five folds'
literature:
- target: '[[viguerasguillen2020dsaek]]'
  relation: source-of-method
- target: '[[viguerasguillen2019fcn]]'
  relation: source-of-method
- target: '[[viguerasguillen2019regression]]'
  relation: explains
  note: 'the group''s edge-map regressor: ECD 4.32 %, CV 7.07 %, HEX 11.74 % MAPE on unhealthy endothelium'
---
# Use the public Vigueras-Guillen weights (on the pod, untested)

## Hypothesis

Weights trained for the Rotterdam papers as a detector or segmenter (GPL-3.0, about 1.04 um/px vs 0.7716).

## Notes

dec-014 closes zero-shot-mask routes for CV/HEX because masks miss cells; an in-domain endothelium segmenter is a different input, but the root-cause oracle bound still applies.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Closed for CV: zero or worse on the stack (CV lane notes 2026-11-11 03:15 EST); the value is not recorded.

## Update 2026-11-13

Numbers now on record: CV +0.0196, HEX +0.0003 (chain, five folds). Closed for both metrics. The PyTorch port of the TVST CNN-Edge / ROI weights (GPL-3.0, about 57 MB) is kept as closed, reusable.
