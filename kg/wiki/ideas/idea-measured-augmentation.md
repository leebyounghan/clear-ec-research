---
id: idea-measured-augmentation
type: idea
name: Augmentation from measured noise and optics (plan G1)
summary: Augment within the measured range of real frames.
status: closed
origin: claude
aims_at:
- CD
- CV
source: docs/PLAN_20261106.md row G1
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-model-reads-grain]]'
builds_on:
- '[[method-measured-augmentation]]'
tested_by:
- target: '[[exp-gpu0-measured-augmentation]]'
  closeness: same
  note: fold 0 only; listed closed in LANES
- target: '[[exp-hist-augmentation-bundle]]'
  closeness: close
- target: '[[exp-cv-dinov3-aug-bundle]]'
  closeness: close
  note: 'running: DINOv3 with the rotation/gamma/blur/noise bundle (not the measured-range transfer)'
---
# Augmentation from measured noise and optics (plan G1)

## Hypothesis

Augment within the measured range of real frames.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
