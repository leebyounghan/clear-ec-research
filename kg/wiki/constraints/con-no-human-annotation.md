---
id: con-no-human-annotation
type: constraint
name: No human annotation
summary: Manual annotation is impossible; the cell-level bridge between image and regression must be built without human labels.
status: active
origin: user
date: '2026-11-03'
source: docs/BRIEF.md Task table; memory/clear-ec-goal-first-place-no-annotation.md
constrains:
- '[[idea-segmenter-self-training]]'
- '[[idea-device-formula-audit]]'
literature:
- target: '[[archit2025microsam]]'
  relation: explains
  note: micro-sam's gains come from annotation-based fine-tuning, which this rule forbids
---
# No human annotation

## Rule

Manual annotation is impossible; the cell-level bridge between image and regression must be built without human labels.
