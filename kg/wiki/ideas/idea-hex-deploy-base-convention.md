---
id: idea-hex-deploy-base-convention
type: idea
name: 'HEX deploy base convention: chain convention against an OOF-input base'
summary: The HEX ridge reads 9.72662 under the chain convention and 9.74136 under an OOF-input base; the deploy fit uses the CV lane's all-rows BEST in-sample as the chain-convention analogue, and the effect of that choice on test is unmeasured.
status: open
origin: claude
aims_at:
- HEX
owner_lane: hex
cost: not estimated in the sources; unmeasurable on OOF by construction
potential_metric: HEX
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: docs/lane_summaries_20261113/final_summary_hex_20261113.md sections 4c and 6; docs/FINAL_SUMMARY_20261113.md section 3.3; shared notes ledger_20261111.jsonl L5, L6 (lines 6-7)
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[con-lesson-l5-deploy-equals-evaluated]]'
- '[[con-lesson-l6-like-for-like]]'
- '[[exp-hex-strict-nesting-audit]]'
tested_by:
- target: '[[exp-hex-strict-nesting-audit]]'
  closeness: related
  note: strict values per convention for the HEX candidates
- target: '[[exp-hex-joint-nested-ridge]]'
  closeness: related
  note: the frozen ridge is quoted in the chain convention (strict 9.71977)
- target: '[[exp-hex-cellroute-step-on-tiers]]'
  closeness: related
  note: 'the cell-route step read under both conventions: (a) chain 9.71943 / 9.73076, (b) OOF base 9.71785 / 9.72863'
---

# HEX deploy base convention: chain convention against an OOF-input base

The HEX ridge reads 9.72662 under the chain convention and 9.74136 under an OOF-input base; the deploy fit uses the CV lane's all-rows BEST in-sample as the chain-convention analogue, and the effect of that choice on test is unmeasured.

## Hypothesis

At deploy the base on the fit rows is the base model fitted on those rows (in-sample: G + RID, r 0.990 with the OOF BEST). Which convention the hidden test follows cannot be read from OOF; the two conventions differ by 0.0147 on the 38-column ridge (arithmetic from 9.74136 - 9.72662).

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Risks

A convention mix-up produced a withdrawn 'joint ridge not robust' message on 11-11 (lesson L6).

## Outcome 2026-11-13

Open at close-out: unverified on test.
