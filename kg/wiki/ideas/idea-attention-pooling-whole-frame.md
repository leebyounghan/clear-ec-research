---
id: idea-attention-pooling-whole-frame
type: idea
name: Attention pooling on the whole frame
summary: Learned attention over tokens instead of std pooling.
status: adopted
origin: claude
aims_at:
- all
owner_lane: backbone
cost: ~12 GPU-h (folds 0,4 under recipe A)
cost_hours: 12
cost_unit: GPU-h
potential_metric: CV
potential_gain: 0.005
potential_basis: 'estimate: window attention over graph windows was closed for CV; frozen pooling variants added -0.0085 to -0.011 as corrections, already in the CV stack'
source: docs/ASSUMPTION_AUDIT_20261106.md 'Model families' (never tried)
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-std-pooling-beats-mean]]'
builds_on:
- '[[method-recipe-a]]'
tested_by:
- target: '[[exp-bb-attention-pool-dinov3]]'
  closeness: same
  note: 'end-to-end 4-query attention pool on DINOv3-B: member equal to plain (9.214 vs 9.219, folds 0,1,2,4); weight 0 in CD and CV'
- target: '[[exp-hex-joint-nested-ridge]]'
  closeness: close
  note: 'its HEX column is the 39th ridge column: strict 9.72662 -> 9.71977'
- target: '[[exp-cv-window-attention]]'
  closeness: related
  note: attention over graph windows, closed for CV
- target: '[[exp-gpu2-frozen-pooling-heads]]'
  closeness: related
  note: frozen heads, per-window outputs
- target: '[[exp-hex-deploy-tiers]]'
  closeness: close
  note: removing the column from tier t2 costs +0.0080 strict; t3 ships without it
---
# Attention pooling on the whole frame

## Hypothesis

Learned attention over tokens instead of std pooling.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Adopted for HEX only, as one column of the joint ridge (D-HEX-004). Closed for CD and CV (weight 0).

## Update 2026-11-13

Still adopted for HEX only: tiers p, t1, t2 and full load the column, t3 does not. CV weight 0 on folds 0+4; CD +0.0047 added / +0.0040 replacing. Open: the column rests on one seed ([[idea-attention-pool-seed-audit]]).
