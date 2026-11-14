---
id: idea-shading-correction
type: idea
name: Flat-field / shading correction before the encoder
summary: Remove vignetting and illumination non-uniformity (2023-24 frames are darker and vignetted) before regression.
status: closed
origin: literature
aims_at:
- CV
- HEX
owner_lane: cd
cost: ~2 CPU-h (flat-field) + ~4 GPU-h member
cost_hours: 4
cost_unit: GPU-h
potential_metric: CD
potential_gain: 0.003
potential_basis: 'estimate: vignetting |r| 0.10 with the CD residual in source B (largest frame statistic); tellez2019augnorm finds normalisation optional when augmenting'
source: paper notes peng2017basic, stringer2025cellpose3
novelty_check: 'KG check 2026-11-09: no experiment corrects shading; the HEX image-statistics ridge (kept) only adds frame statistics; measured augmentation transfers grey levels. Web search not yet done.'
motivated_by:
- '[[find-recent-images-darker]]'
builds_on:
- '[[method-recipe-a]]'
tested_by:
- target: '[[exp-bb-shading-correction]]'
  closeness: same
  note: 'under leave-one-source-out the source offset rose 0.0825 -> 0.0959; closed'
- target: '[[exp-hex-image-stats-ridge]]'
  closeness: related
  note: frame statistics as features, kept on HEX
- target: '[[exp-gpu0-measured-augmentation]]'
  closeness: related
- target: '[[exp-gpu0-frame-quality-vs-residual]]'
  closeness: related
  note: vignette |r| 0.10 with CD residual in source B, the largest frame statistic
literature:
- target: '[[peng2017basic]]'
  relation: source-of-method
- target: '[[stringer2025cellpose3]]'
  relation: explains
  note: Cellpose3 restoration does not model vignetting
- target: '[[karani2021tta]]'
  relation: suggests
  note: a tiny learned input-normalisation module in front of a frozen backbone
- target: '[[kolluru2019unet]]'
  relation: corroborates
  note: illumination flattening mattered for classical binarisation
- target: '[[tellez2019augnorm]]'
  relation: contradicts
  note: normalisation optional, augmentation the must-have
- target: '[[yoon2024dgreview]]'
  relation: suggests
  note: histogram matching as a cheap baseline
---
# Flat-field / shading correction before the encoder

## Hypothesis

Remove vignetting and illumination non-uniformity (2023-24 frames are darker and vignetted) before regression.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).

## Outcome 2026-11-11

Closed by the backbone lane (notes 2026-11-11 05:15 EST): shading correction raised the cross-source level offset from 0.0825 to 0.0959 ([[find-source-shift-level-offset]]).
