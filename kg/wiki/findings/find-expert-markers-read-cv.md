---
id: find-expert-markers-read-cv
type: finding
name: Expert markers read CV at r 0.89, MAPE 8.8 on 25 frames
summary: 'Device values read from the 25 expert-marker sets: CV r 0.88-0.90 (area CV r 0.89, LOO MAPE 8.8 after scaling), HEX r 0.84, CD r 0.75. Hybrid B on the same frames: HEX 0.81, CV 0.53, CD 0.77.'
status: robust
metrics:
- CV
- HEX
- CD
date: '2026-11-08'
source: 'docs/BRIEF.md ''Reading device values from expert markers''; experiments/rootcause_20261109/out/q1_seg_loose.json (oracle rows: CV r 0.882, LOO MAPE 8.844)'
---

# Expert markers read CV at r 0.89, MAPE 8.8 on 25 frames

Device values read from the 25 expert-marker sets: CV r 0.88-0.90 (area CV r 0.89, LOO MAPE 8.8 after scaling), HEX r 0.84, CD r 0.75. Hybrid B on the same frames: HEX 0.81, CV 0.53, CD 0.77.

## Limits and notes

25 frames, one 2023 style. The gap between hybrid B (0.53) and markers (0.89) on CV is the cell-reading gap; on HEX the pipeline is already close.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
