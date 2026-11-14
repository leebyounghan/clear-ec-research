---
id: idea-fourier-ring-cd-column
type: idea
name: A label-free Fourier ring estimate of cell spacing as a CD column
summary: The classic frequency-analysis estimate of endothelial density (radial power-spectrum ring, CD ~ 1/spacing^2) as a label-free CD column in the mask slot or as a member.
status: closed
origin: claude
aims_at:
- CD
owner_lane: deploy
cost: 108 s on 15 CPU workers for all frames, plus one CPU stack read
cost_hours: 0.5
cost_unit: CPU-h
potential_metric: CD
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: shared notes ledger_20261111.jsonl D-CDSIMPLE-001 (line 91), A-DEPLOY-005 (line 98), O-CD-FFT-001 (line 106)
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[exp-ref-submission1]]'
- '[[find-mask-cd-direct-density]]'
tested_by:
- target: '[[exp-deploy-fourier-ring-cd-alone]]'
  closeness: same
  note: 'alone: log r 0.676, nested MAPE 9.59 (cd_mm2 7.37); deviation from the stack does not predict its error (r 0.021)'
- target: '[[exp-cd-fourier-ring-cd-stack]]'
  closeness: same
  note: weight 0 in every fold, strict five folds; in place of cd_mm2 +0.0155
- target: '[[exp-hist-cd-density-channel]]'
  closeness: close
  note: Fourier spacing + detector density as a CD residual channel, closed before the lanes with no number recorded
uses_method:
- '[[method-fourier-ring-density]]'
---

# A label-free Fourier ring estimate of cell spacing as a CD column

The classic frequency-analysis estimate of endothelial density (radial power-spectrum ring, CD ~ 1/spacing^2) as a label-free CD column in the mask slot or as a member.

## Hypothesis

Named by the coordinator after the PI said there is surely a simple, non-tricky way to lower CD (D-CDSIMPLE-001 a).

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-13

Closed for CD: weight 0 in the stack. The deploy lane's summary lists it as open only because that lane had not seen the CD lane's stack read.
