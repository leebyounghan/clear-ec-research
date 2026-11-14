---
id: idea-device-formula-audit
type: idea
name: Audit the device formulas on the 25 marker sets (plan C2)
summary: Check how the device turns cells into CD, CV, HEX (e.g. CV ~2x Voronoi area CV) before trusting the marker table.
status: closed
origin: claude
aims_at:
- CV
- HEX
source: docs/PLAN_20261106.md row C2
novelty_check: 'KG check 2026-11-09: answered by exp-cv-konan-bisectors-markers (CV lane line 85); CD formula known (find-cd-label-definition). Literature: takahashi1996konanpatent, gasser2015konantopcon, bucht2011simulation.'
motivated_by:
- '[[find-device-cv-twice-voronoi]]'
- '[[find-residual-at-oracle-floor-cv-hex]]'
builds_on:
- '[[method-expert-marker-oracle]]'
tested_by:
- target: '[[exp-rc-variance-budget]]'
  closeness: related
  note: '78 % (CV) / 88 % (HEX) of our residual variance is what expert centres miss: device behaviour plus label noise'
- target: '[[exp-cv-konan-bisectors-markers]]'
  closeness: close
  note: 'the audit''s question answered for CV and HEX: Konan weighted bisectors (k=2) reproduce the device scale (1.09 vs 2.08 for Voronoi) on the 25 marker sets'
literature:
- target: '[[bucht2011simulation]]'
  relation: suggests
  note: simulator with exact morphometry found the Topcon CV scaling error
- target: '[[piorkowski2017segmentation]]'
  relation: suggests
  note: segmentation method changes hexagonality more than CV
---
# Audit the device formulas on the 25 marker sets (plan C2)

## Hypothesis

Check how the device turns cells into CD, CV, HEX (e.g. CV ~2x Voronoi area CV) before trusting the marker table.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-09

Answered: the device's CV and HEX follow the Konan centre-method construction with weighted bisectors (k=2), see [[find-konan-rule-explains-device-cv]]. Status set to closed (audit done).
