---
id: find-screen-folds-overstate
type: finding
name: Folds 0 and 4 (and single folds) overstate gains that later shrink on five folds
summary: ConvNeXt on CD -0.045 on folds 0,4 -> -0.002 beyond DINOv3 on five; DINOv3 on CV -0.031 -> -0.009; 50-epoch graphs -0.027 -> -0.007; RoPE first look overstated threefold; CPSAM fold 4 -0.0127 -> two folds -0.0006.
status: robust
metrics:
- CD
- CV
- HEX
- mean
date: '2026-11-08'
source: lane RESULTS and REPORTs (see supporting experiments); memory/clear-ec-coordinator-audit-lessons.md; shared notes ledger_20261111.jsonl O-CV-SYN4-001 (line 122), O-CD-SYN5-READ (line 178), O-HEX-008 (line 112)
---

# Folds 0 and 4 (and single folds) overstate gains that later shrink on five folds

ConvNeXt on CD -0.045 on folds 0,4 -> -0.002 beyond DINOv3 on five; DINOv3 on CV -0.031 -> -0.009; 50-epoch graphs -0.027 -> -0.007; RoPE first look overstated threefold; CPSAM fold 4 -0.0127 -> two folds -0.0006.

## Limits and notes

Read five folds before stacking more of a family; never call a one- or two-fold result a success (con-name-baseline-and-folds).

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.

## Update 2026-11-13

Three more folds 0+4 reads that did not hold on five folds, strict, incumbent in the same run: synthetic ViT-L in the CV stack -0.0154 -> +0.0078 ([[exp-cv-synth-vitl-stack]]); synthetic recipe A in the CD stack -0.0059 -> -0.0017 ([[exp-cd-synth-recipe-a-stack]]); synthetic recipe A in the HEX ridge -0.00335 -> +0.0017 ([[exp-hex-synth-recipe-a-swap]]).
