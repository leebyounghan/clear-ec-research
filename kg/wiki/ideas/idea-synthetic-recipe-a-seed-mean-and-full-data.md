---
id: idea-synthetic-recipe-a-seed-mean-and-full-data
type: idea
name: Synthetic recipe A seed mean on folds 1-4, and full-data synthetic builds
summary: The seed mean that L3 requires for an adoption read of synthetic-pretrained recipe A exists on fold 0 only, and no full-data synthetic build was completed; neither was run because the routes were closed.
status: partially-tested
origin: claude
aims_at:
- CD
owner_lane: backbone
cost: about 8.1 h of fine-tuning per fold and seed on one GPU with the shared pretraining; a seed-47 pretraining replicate adds about 2.5 h
cost_hours: 32
cost_unit: GPU-h
potential_metric: CD
potential_basis: 'not estimated: no source gives an estimate of the gain'
source: shared notes ledger_20261111.jsonl A-BB-SYN7-RESULT (line 191), O-BB-GPU013-IDLE (line 180), D-SYN-006 (line 136), D-SYN-008 (line 181); docs/lane_summaries_20261113/final_summary_backbone_20261113.md section 4c; docs/lane_summaries_20261113/final_summary_cd_20261113.md section 4c; docs/FINAL_SUMMARY_20261113.md section 3.3
novelty_check: 'KG ingest 2026-11-13: recorded after the fact from the shared ledger and the lane final summaries; kg_query check and a grep over wiki/ found no earlier node for it; no web search in this pass.'
motivated_by:
- '[[exp-cd-synth-recipe-a-stack]]'
- '[[con-lesson-l3-seed-mean-before-adoption]]'
tested_by:
- target: '[[exp-bb-synth-recipe-a-seed47-fold0]]'
  closeness: close
  note: 'fold 0 only: seed mean -0.022 (CI -0.057 to +0.016), smaller than the plain seed-to-seed spread of +0.042'
- target: '[[exp-bb-synth-vitl]]'
  closeness: related
  note: the full-data synthetic ViT-L build was stopped at epoch 9 of 12 (D-SYN-006), archived and never verified
depends_on:
- '[[idea-synthetic-specular-pretraining]]'
---

# Synthetic recipe A seed mean on folds 1-4, and full-data synthetic builds

The seed mean that L3 requires for an adoption read of synthetic-pretrained recipe A exists on fold 0 only, and no full-data synthetic build was completed; neither was run because the routes were closed.

## Hypothesis

The five-fold strict CD read of synthetic recipe A (-0.0017) rests on one fine-tune seed; a seed mean on all folds would say whether anything is there.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-13

Not run: the backbone lane's offer to run seed 47 on folds 1-4 was withdrawn when the route was closed (D-SYN-008) and the GPUs were then taken down (D-GPU-002). The member read is null, so this kept an option open and was never a lead.
