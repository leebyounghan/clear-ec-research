---
id: find-recent-images-darker
type: finding
name: 2023-24 images are darker and vignetted, with higher CV and lower HEX
summary: Acquisition era shifts image appearance and label distributions.
status: robust
metrics:
- CV
- HEX
date: '2026-11-08'
source: 'docs/BRIEF.md ''Measured facts: Labels'''
literature:
- target: '[[peng2017basic]]'
  relation: suggests
  note: retrospective flat-field from many frames (global or per image; per-source keys are not allowed)
- target: '[[yoon2024dgreview]]'
  relation: explains
  note: data-level normalisation/augmentation as the first step against acquisition shift
- target: '[[tellez2019augnorm]]'
  relation: suggests
  note: appearance augmentation over normalisation
---
# 2023-24 images are darker and vignetted, with higher CV and lower HEX

Acquisition era shifts image appearance and label distributions.

## Limits and notes

HEX bias by imaging year -1.3 % (2021) to +0.3 % (2023) (HEX REPORT); image statistics stand in for year in the HEX chain.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
