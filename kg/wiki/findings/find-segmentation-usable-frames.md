---
id: find-segmentation-usable-frames
type: finding
name: Loose thresholds + CLAHE make 85-90 % of frames segmentable (default 53 %)
summary: Zero-shot Cellpose-SAM, flow 0.8, cellprob -2.
status: robust
metrics:
- HEX
- CD
date: '2026-11-08'
source: experiments/lane_hex_20261108/REPORT.md 'Segmentation'
literature:
- target: '[[pachitariu2025cellposesam]]'
  relation: explains
---

# Loose thresholds + CLAHE make 85-90 % of frames segmentable (default 53 %)

Zero-shot Cellpose-SAM, flow 0.8, cellprob -2.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
