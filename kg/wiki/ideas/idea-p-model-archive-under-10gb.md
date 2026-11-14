---
id: idea-p-model-archive-under-10gb
type: idea
name: Bring the tier p model archive under 10 GB per file
summary: Move about 5.5-6.3 GB of weights from the model archive into the image, or add more fp16 weight copies, so that both upload files are under a possible 10 GB cap. Not tested.
status: never-tried
origin: claude
aims_at:
- all
owner_lane: deploy
cost: a restage, a rebuild and the packaged-mode parity runs again; not estimated in hours
potential_metric: all
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: shared notes ledger_20261111.jsonl F-DEPLOY-006 (line 201); docs/lane_summaries_20261113/final_summary_deploy_20261113.md section 4c; docs/FINAL_SUMMARY_20261113.md section 3.3
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[find-upload-size-risk-p-model-archive]]'
- '[[exp-deploy-tier-table]]'
tested_by:
- target: '[[exp-deploy-packaged-mode-parity]]'
  closeness: related
  note: the parity runs that any new fp16 copy would have to repeat
- target: '[[exp-deploy-buildbox-container-test-t3]]'
  closeness: related
  note: 'the fallback already under the cap: t3 model archive 8.81 GB'
---

# Bring the tier p model archive under 10 GB per file

Move about 5.5-6.3 GB of weights from the model archive into the image, or add more fp16 weight copies, so that both upload files are under a possible 10 GB cap. Not tested.

## Hypothesis

Tier p's model archive is 15.06 GB against a possible 10 GB per-upload cap. Moving weights into the image would leave both files at about 9.4-10 GB gz (tight) and the image at about 14 GB uncompressed against a stated 10 GB container guideline; more fp16 copies change the numerics and need the parity runs again.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Risks

Both variants reopen a built and tested package the day before the upload.

## Outcome 2026-11-13

Not done by the deploy lane. The standing rule is to upload p as planned and switch both files to t3 if the model upload is refused ([[dec-020-t4-assumed-tier-p-t3-fallback]]).
