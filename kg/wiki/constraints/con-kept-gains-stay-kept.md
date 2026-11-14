---
id: con-kept-gains-stay-kept
type: constraint
name: Kept gains stay kept
summary: 'Steps already kept stay, however small; ''no detail chasing'' decides where new time goes, never removes a kept gain. Corrected 2026-11-09: the coordinator had added ''prefer simple corrections over many small steps'', which was not the PI''s rule.'
status: active
origin: user
date: '2026-11-09'
source: docs/LANES_20261108.md 'Bottleneck first'; memory/clear-ec-bottleneck-timebox-no-detail.md; memory/clear-ec-coordinator-audit-lessons.md
constrains:
- '[[dec-005-kept-gains-correction]]'
---

# Kept gains stay kept

## Rule

Steps already kept stay, however small; 'no detail chasing' decides where new time goes, never removes a kept gain. Corrected 2026-11-09: the coordinator had added 'prefer simple corrections over many small steps', which was not the PI's rule.
