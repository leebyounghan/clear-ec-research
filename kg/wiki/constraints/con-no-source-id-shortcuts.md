---
id: con-no-source-id-shortcuts
type: constraint
name: No shortcuts on source, ID format, donor or fold
summary: Do not let a model or correction key on image source or ID format, file names, donor or fold identity, or a handful of odd rows; make no per-source or per-fold choices.
status: active
origin: user
date: '2026-11-08'
source: docs/LANES_20261108.md 'Bottleneck first'; memory/clear-ec-bottleneck-timebox-no-detail.md
constrains:
- '[[method-per-source-correction]]'
- '[[idea-exclude-hyphen-cv-from-loss]]'
- '[[exp-cd-fellow-eye-residual-transfer]]'
---

# No shortcuts on source, ID format, donor or fold

## Rule

Do not let a model or correction key on image source or ID format, file names, donor or fold identity, or a handful of odd rows; make no per-source or per-fold choices.

## Applied 2026-11-13

The PI closed a fellow-eye residual transfer as a trick, not to be used ([[exp-cd-fellow-eye-residual-transfer]]): its oracle form finds the partner eye through the frame ID (CD -0.236 on tier p), and the image-only form gains nothing.
