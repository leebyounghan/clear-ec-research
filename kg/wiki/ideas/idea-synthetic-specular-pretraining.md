---
id: idea-synthetic-specular-pretraining
type: idea
name: Pretraining on synthetic specular frames with exact labels
summary: Rendered mosaics with exact CD, area CV and HEX give label-noise-free supervision before fine-tuning on the noisy device labels.
status: closed
origin: claude
aims_at:
- all
owner_lane: backbone
source: shared notes ledger_20261111.jsonl D-SYN-001, D-SYN-002, D-SYN-003; lane_backbone_notes.md; D-SYN-004 to D-SYN-008 (ledger lines 81, 131, 136, 164, 181); docs/FINAL_SUMMARY_20261113.md section 3.2
novelty_check: 'KG ingest 2026-11-11: recorded after the fact from the shared ledger and lane notes; no separate KG or web search was run for this node.'
motivated_by:
- '[[find-lit-cv-hex-grader-noise]]'
- '[[find-residual-at-oracle-floor-cv-hex]]'
tested_by:
- target: '[[exp-bb-synth-zeroshot]]'
  closeness: close
  note: 'zero-shot columns: closed in all three lanes (HEX +0.00109)'
- target: '[[exp-bb-synthft3-dinov3]]'
  closeness: same
  note: DINOv3-B single member -0.072 on five folds; not adopted in stacks (CV swap +0.0035 strict)
- target: '[[exp-bb-synth-vitl]]'
  closeness: close
  note: 'ViT-L s46, five folds: member -0.068; closed (D-SYN-006)'
- target: '[[exp-bb-synth-recipe-a]]'
  closeness: close
  note: 'recipe A s46, five folds: member -0.0063 (CI -0.0293 to +0.0160); closed (D-SYN-008)'
- target: '[[exp-cv-synth-vitl-stack]]'
  closeness: same
  note: 'CV stack, strict five folds: +0.0078 (folds 0+4 screen -0.0154)'
- target: '[[exp-cd-synth-vitl-stack]]'
  closeness: same
  note: 'CD stack, strict five folds: -0.0031, folds mixed, under seed noise'
- target: '[[exp-cd-synth-recipe-a-stack]]'
  closeness: same
  note: 'CD stack, strict five folds: -0.0017 (CI -0.012 to +0.008)'
- target: '[[exp-cv-synth-recipe-a-swap]]'
  closeness: same
  note: 'CV stack, strict five folds: +0.0041'
- target: '[[exp-hex-synth-recipe-a-swap]]'
  closeness: same
  note: 'HEX ridge, strict five folds: +0.0017'
- target: '[[exp-bb-synth-recipe-a-seed47-fold0]]'
  closeness: close
  note: 'second fine-tune seed, fold 0 only: seed mean -0.022 (CI -0.057 to +0.016)'
cost: 'spent: three routes on five folds each (recipe A about 2.5 h pretraining + 8.1 h fine-tuning per fold)'
potential_metric: CD
uses_method:
- '[[method-synthetic-specular-pretraining]]'
builds_on:
- '[[ds-simulated-tessellations]]'
---

# Pretraining on synthetic specular frames with exact labels

## Hypothesis

Rendered mosaics with exact CD, area CV and HEX give label-noise-free supervision before fine-tuning on the noisy device labels.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Single DINOv3-B model -0.072 on five folds; not adopted in any stack. The ViT-L and recipe A arms are running outside the freeze ([[dec-017-synthetic-pretraining-followup]]).

## Update 2026-11-13

Closed on all three routes and all three metrics: member-level gains (-0.072 DINOv3-B, -0.068 ViT-L, -0.006 recipe A) vanish in the stacks ([[find-synthetic-pretraining-gains-vanish-in-stacks]]). Decisions D-SYN-001, D-SYN-006 (PI) and D-SYN-008 in [[dec-017-synthetic-pretraining-followup]]. Not run: the seed mean on folds 1-4 and any full-data synthetic build ([[idea-synthetic-recipe-a-seed-mean-and-full-data]]). The 63 GB synthetic set is [[ds-synthetic-specular-v3]].
