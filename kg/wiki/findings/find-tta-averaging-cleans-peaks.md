---
id: find-tta-averaging-cleans-peaks
type: finding
name: Averaging detector heatmaps over flips and seeds gives cleaner peaks and better CV graphs
summary: '25 frames: false 0.130 -> 0.107, F1 0.826 -> 0.838; on all labels the gain comes from averaging, not the threshold (control +0.004).'
status: robust
metrics:
- CV
date: '2026-11-08'
source: experiments/lane_cv_20261108/RESULTS.tsv
literature:
- target: '[[wang2019tta]]'
  relation: corroborates
  note: TTA averaging reduces variance of dense predictions
---
# Averaging detector heatmaps over flips and seeds gives cleaner peaks and better CV graphs

25 frames: false 0.130 -> 0.107, F1 0.826 -> 0.838; on all labels the gain comes from averaging, not the threshold (control +0.004).

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
