---
id: idea-cd-recalibration-against-earlier-models
type: idea
name: Recalibrate the CD candidate, or blend in submission 1 or the Phase I best model
summary: If an earlier model were better calibrated at the ends of the CD range, recalibrating the candidate against it or blending it in would lower CD at no GPU cost.
status: closed
origin: user
aims_at:
- CD
owner_lane: cd
cost: CPU only
cost_hours: 1
cost_unit: CPU-h
potential_metric: CD
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: shared notes ledger_20261111.jsonl O-CD-SUB1-CALIB (line 204), O-CD-PHASE1-CALIB (line 205), F-CDANAT-001 (line 92)
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[exp-ref-submission1]]'
- '[[find-cv-error-in-tails]]'
tested_by:
- target: '[[exp-cd-sub1-calibration-read]]'
  closeness: same
  note: recalibrations +0.004 to +0.050; blend +0.0024 (+0.0011 with stretch); submission 1 is spanned by candidate members
- target: '[[exp-cd-phase1-calibration-read]]'
  closeness: same
  note: 'folds 2-4: blend +0.0025 (CI -0.003 to +0.009); the sixth member -0.0005, not significant'
- target: '[[exp-cd-cheap-posthoc-levers]]'
  closeness: close
  note: global / per-level multipliers +0.0046 / +0.0050, clipping +0.0384 on the 11-11 strict OOF
- target: '[[exp-cd-uncertainty-shift]]'
  closeness: related
  note: member-disagreement MAPE shift, +0.012 to +0.025, closed on 11-08
---

# Recalibrate the CD candidate, or blend in submission 1 or the Phase I best model

If an earlier model were better calibrated at the ends of the CD range, recalibrating the candidate against it or blending it in would lower CD at no GPU cost.

## Hypothesis

PI request of 2026-11-13 21:30 EST (submission 1) and correction of 21:33 EST (redo with the Phase I best model).

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-13

Nothing there: [[find-cd-no-cheap-posthoc-lever]]. The regression-to-the-mean pattern by label decile is what conditioning on a noisy label produces; a function of the prediction cannot remove it.
