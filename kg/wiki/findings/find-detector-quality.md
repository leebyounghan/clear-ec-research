---
id: find-detector-quality
type: finding
name: 'Pre-only detector on the 25 marker frames: miss 0.150, false 0.165, jitter 0.215, F1 0.862'
summary: The deployed pre-only detector (trained on the external Topcon set).
status: robust
metrics:
- CV
- HEX
date: '2026-11-08'
source: docs/BRIEF.md 'Detector on the 25 sets'; docs/ASSUMPTION_AUDIT_20261106.md belief 11
---

# Pre-only detector on the 25 marker frames: miss 0.150, false 0.165, jitter 0.215, F1 0.862

The deployed pre-only detector (trained on the external Topcon set).

## Limits and notes

F1 is a poor proxy for hex6 gain (audit belief 11). Fusing detectors barely improves points: errors are shared.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
