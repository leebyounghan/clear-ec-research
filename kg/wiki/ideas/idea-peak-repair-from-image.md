---
id: idea-peak-repair-from-image
type: idea
name: Peak repair from image evidence (plan C4)
summary: Recover missed and remove false peaks using image evidence before the graph.
status: partially-tested
origin: claude
aims_at:
- CV
- HEX
owner_lane: cv
cost: ~4 CPU-h (repair on stored heatmaps + Konan/graph re-read)
cost_hours: 4
cost_unit: CPU-h
potential_metric: CV
potential_gain: 0.005
potential_basis: 'estimate: large in principle (expert centres with the Konan rule read CV LOO-MAPE 7.64 on 25 frames) but spacing-prior selection and graph repair are closed and 16-detector fusion barely improves points (miss 0.20 vs 0.22)'
source: docs/PLAN_20261106.md row C4; docs/ASSUMPTION_AUDIT_20261106.md belief 11
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-detector-misses-ruin-area-cv]]'
- '[[find-detector-quality]]'
- '[[find-konan-rule-explains-device-cv]]'
builds_on:
- '[[method-pre-only-detector]]'
tested_by:
- target: '[[exp-cv-spacing-prior-nms]]'
  closeness: close
  note: spacing-prior selection and gap fill, closed
- target: '[[exp-hist-graph-repair]]'
  closeness: close
  note: graph repair closed for HEX
- target: '[[exp-cv-tta-heatmap-graphs]]'
  closeness: related
  note: averaging heatmaps cleans peaks, kept
literature:
- target: '[[viguerasguillen2022feedbackattn]]'
  relation: suggests
  note: a cell-body probability head to select trustworthy cells (CD/selection use only after dec-014)
---
# Peak repair from image evidence (plan C4)

## Hypothesis

Recover missed and remove false peaks using image evidence before the graph.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
