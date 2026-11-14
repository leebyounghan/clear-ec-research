---
id: find-synthetic-pretraining-gains-vanish-in-stacks
type: finding
name: Synthetic pretraining improves members alone (-0.072, -0.068, -0.006) but the gains vanish in the stacks
summary: 'Member reads, five folds, one seed, against the plain model: DINOv3-B -0.072, ViT-L -0.068, recipe A -0.006. In the stacks, strict five folds with the incumbent in the same run: CD -0.0031 (ViT-L) and -0.0017 (recipe A), both inside seed noise with mixed fold signs; CV +0.0035 (DINOv3-B), +0.0041 (recipe A), +0.0078 (ViT-L); HEX +0.0010 (DINOv3-B) and +0.0017 (recipe A). All three routes are closed.'
status: robust
metrics:
- CD
- CV
- HEX
- mean
date: '2026-11-13'
source: shared notes ledger_20261111.jsonl D-SYN-001 (line 13), D-SYN-006 (line 136), D-SYN-008 (line 181), O-CV-SYN-002 (line 59), O-HEX-003 (line 20), O-CV-SYN4-001 (line 122), O-CD-SYN4-001 (line 125), O-CV-SYN3-001 (line 113), O-HEX-008 (line 112), O-CD-SYN5-READ (line 178), A-BB-SYN5-READ (line 179), A-BB-SYNVITL-F3 (line 123); docs/FINAL_SUMMARY_20261113.md section 3.2
motivated_by:
- '[[find-better-alone-not-stack-weight]]'
evidence_for:
- '[[dec-017-synthetic-pretraining-followup]]'
---

# Synthetic pretraining improves members alone (-0.072, -0.068, -0.006) but the gains vanish in the stacks

Member reads, five folds, one seed, against the plain model: DINOv3-B -0.072, ViT-L -0.068, recipe A -0.006. In the stacks, strict five folds with the incumbent in the same run: CD -0.0031 (ViT-L) and -0.0017 (recipe A), both inside seed noise with mixed fold signs; CV +0.0035 (DINOv3-B), +0.0041 (recipe A), +0.0078 (ViT-L); HEX +0.0010 (DINOv3-B) and +0.0017 (recipe A). All three routes are closed.

## Evidence

| Route | Member alone, five folds | CD stack, strict | CV stack, strict | HEX, strict |
|---|---|---|---|---|
| DINOv3-B ([[exp-bb-synthft3-dinov3]]) | 9.1606 vs 9.2326 (-0.072) | weight 0 (folds 0+4) | +0.0035 | +0.00101 (member on folds 0+4 only) |
| ViT-L ([[exp-bb-synth-vitl]]) | 9.1654 vs 9.2330 (-0.068) | -0.0031, folds mixed ([[exp-cd-synth-vitl-stack]]) | +0.0078 ([[exp-cv-synth-vitl-stack]]) | not an input of the ridge |
| Recipe A ([[exp-bb-synth-recipe-a]]) | 9.0198 vs 9.0261 (-0.0063, CI -0.0293 to +0.0160) | -0.0017, CI -0.012 to +0.008 ([[exp-cd-synth-recipe-a-stack]]) | +0.0041 ([[exp-cv-synth-recipe-a-swap]]) | +0.0017 ([[exp-hex-synth-recipe-a-swap]]) |

The folds 0+4 screens had read CV -0.0110 (DINOv3-B) and -0.0154 (ViT-L) and CD -0.0059 (recipe A); none held on five folds. Zero-shot use of the synthetic model was closed earlier ([[exp-bb-synth-zeroshot]]).

## Limits

One fine-tune seed per route; the only second seed is recipe A on fold 0 ([[exp-bb-synth-recipe-a-seed47-fold0]]: seed mean -0.022, CI -0.057 to +0.016). The pretraining stage is shared between seeds. The full-data synthetic ViT-L build was stopped at epoch 9 of 12 and never verified. One synthetic set (synth_v3) and one pass over it.

## Bears on

Closes [[idea-synthetic-specular-pretraining]] ([[dec-017-synthetic-pretraining-followup]]). It is the L1 pattern again ([[find-better-alone-not-stack-weight]]) and three more screen-fold gains that did not hold ([[find-screen-folds-overstate]]).
