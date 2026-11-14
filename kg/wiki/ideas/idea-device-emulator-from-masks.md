---
id: idea-device-emulator-from-masks
type: idea
name: Emulate the device readout from masks
summary: Compute CD, CV, HEX device-style from segmentation masks and add them to the pipeline.
status: closed
origin: claude
aims_at:
- CV
- HEX
- CD
source: root-cause analysis 2026-11-09
novelty_check: Not yet checked against the literature (2026-11-09); run the idea check in CLAUDE.md before spending GPU time.
motivated_by:
- '[[find-hex-matches-hex6]]'
- '[[find-device-cv-twice-voronoi]]'
builds_on:
- '[[method-mask-emulator]]'
tested_by:
- target: '[[exp-rc-mask-emulator]]'
  closeness: same
  note: CD -0.009, CV ~0, HEX -0.003 to -0.008
- target: '[[exp-hex-seg-shape-ridge]]'
  closeness: related
  note: mask shape ridge kept in the HEX chain (-0.0235)
---

# Emulate the device readout from masks

## Hypothesis

Compute CD, CV, HEX device-style from segmentation masks and add them to the pipeline.

## Closest tries in the KG

See `tested_by` (closeness: same = the same test on the same inputs; close = a clearly related version; related = shares a mechanism).
