---
id: method-target-denoising
type: method
name: 'Target denoising: soft teacher targets and teacher-based row reweighting'
summary: Training targets softened toward a label-free nested teacher (zero-shot Cellpose-SAM masks + image statistics), or rows reweighted by their distance from the teacher.
code: lane_backbone_20261108/source make_teacher*, make_nested_teacher, rowweight_from_teacher
---

# Target denoising: soft teacher targets and teacher-based row reweighting

Training targets softened toward a label-free nested teacher (zero-shot Cellpose-SAM masks + image statistics), or rows reweighted by their distance from the teacher.

## Where it is used

Experiments link here with `uses_method`; see backlinks. Sources: `docs/lane_summaries_20261113/final_summary_backbone_20261113.md` sections 3 and 4 (`lane_backbone_20261108/data/teacher_*.npz`, `rowweight_*.npz`).
