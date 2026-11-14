---
id: idea-cv-fp16-path-impact
type: idea
name: CV-specific score impact of the T4 fp16 path
summary: The T4 fp16 path deviates CV by median 0.175 % (max about 0.9 %); no CV-specific score impact was measured by the CV lane.
status: open
origin: claude
aims_at:
- CV
owner_lane: cv
cost: CPU only from the stored 200-row reports; not estimated in the sources
potential_metric: CV
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: docs/lane_summaries_20261113/final_summary_cv_20261113.md section 4c (idea-cv-fp16-path-impact); shared notes ledger_20261111.jsonl O-DEPLOY-020 (line 185), O-DEPLOY-021 (line 188)
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[find-kernel-change-not-precision-moves-outputs]]'
- '[[exp-deploy-t4-default-200-rows-p]]'
tested_by:
- target: '[[exp-deploy-t4-default-200-rows-p]]'
  closeness: related
  note: score impact +0.0067 (+0.0017 to +0.0122), HEX +0.018; CV deviation median 0.175 %, max 0.83 %
- target: '[[exp-deploy-t4-all-fp32-200-rows]]'
  closeness: related
  note: 'paired fp32 minus fp16 on 200 rows: CV +0.0088'
---

# CV-specific score impact of the T4 fp16 path

The T4 fp16 path deviates CV by median 0.175 % (max about 0.9 %); no CV-specific score impact was measured by the CV lane.

## Hypothesis

The deploy lane's impact is the mean over the three metrics with HEX carrying about 3/4; the CV share alone was not broken out by the CV lane.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-13

Open at close-out.
