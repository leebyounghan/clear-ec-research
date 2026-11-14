---
id: idea-cd-only-loss-recipe-a
type: idea
name: Train recipe A with the CD loss only (and a higher encoder LR)
summary: All image members are trained on CD, CV and HEX jointly; a CD-only objective on recipe A, at encoder LR 1e-5 and 3e-5, tests whether the members are under-trained for CD.
status: closed
origin: user
aims_at:
- CD
owner_lane: cd
cost: about 8 h per fold and arm on one GPU; four fold runs were made
cost_hours: 32
cost_unit: GPU-h
potential_metric: CD
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: shared notes ledger_20261111.jsonl D-CDTRAIN-001 / 002 / 003 (lines 90, 104, 118), O-CDTRAIN-1E5-F04 (line 134), O-CDTRAIN-3E5-F04 (line 128)
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[exp-ref-submission1]]'
- '[[exp-cd-epoch-probe]]'
tested_by:
- target: '[[exp-cd-cdonly-recipe-a-elr1e5]]'
  closeness: same
  note: 'encoder LR 1e-5: -0.0030 on folds 0+4 strict (fold 0 -0.009, fold 4 +0.003); equals plain alone'
- target: '[[exp-cd-cdonly-recipe-a-elr3e5]]'
  closeness: same
  note: 'encoder LR 3e-5: 0.09 better alone, +0.0181 in the stack on folds 0+4 strict'
- target: '[[exp-cd-epoch-probe]]'
  closeness: related
  note: CD single-epoch validation bottoms at epochs 9-12; no support for under-training from the schedule
- target: '[[exp-cd-cd-weighted-loss]]'
  closeness: close
  note: CD-weighted loss (x3) on a family member, planned on 11-09 and never run
uses_method:
- '[[method-metric-weighted-loss]]'
---

# Train recipe A with the CD loss only (and a higher encoder LR)

All image members are trained on CD, CV and HEX jointly; a CD-only objective on recipe A, at encoder LR 1e-5 and 3e-5, tests whether the members are under-trained for CD.

## Hypothesis

PI hypotheses after the 11-11 leaderboard: (H1) CD is the easy target and our members are under-trained for it; (H2) something in our pipeline suppresses CD. Stack-level suppression was already ruled out (from-scratch search 6.1733 vs 6.1741). The coordinator turned H1 into a CD-only loss arm and a longer-schedule arm; the probe showed CD is not falling at the last epoch, so the longer schedule was dropped and a second encoder LR took its place.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-13

Both arms failed the rule fixed before the read (more than 0.01 below the same-run incumbent with both folds lower) and are closed. The joint loss was never varied in any other way ([[idea-recipe-a-hyperparameters-never-varied]]).
