---
id: idea-external-author-access-data
type: idea
name: Train with author-access endothelium data (Rotterdam / Topcon SP-1P, CPTS)
summary: More labelled specular data, ideally with cell segmentations.
status: open
origin: user
aims_at:
- all
owner_lane: coordinator
cost: request time (dec-009) + ~12 GPU-h
cost_hours: 12
cost_unit: GPU-h
potential_metric: all
potential_gain: null
potential_basis: 'unknown: halving labels costs ~0.08 (fold 0) so more data could help, but Topcon CV/HEX labels sit on another device scale (find-lit-topcon-konan-cv-scale); depends on dec-009'
source: docs/BRIEF.md 'Open decisions'; docs/PLAN_20261106.md 'Decisions'
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-label-fraction-cost]]'
builds_on:
- '[[ds-rotterdam-author-access]]'
- '[[ds-cpts-request]]'
depends_on:
- '[[dec-009-author-access-request]]'
literature:
- target: '[[viguerasguillen2020dsaek]]'
  relation: source-of-method
- target: '[[gasser2015konantopcon]]'
  relation: contradicts
  note: Topcon-device CV/HEX labels are on another scale (CV 37 vs 29)
---
# Train with author-access endothelium data (Rotterdam / Topcon SP-1P, CPTS)

## Hypothesis

More labelled specular data, ideally with cell segmentations.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
