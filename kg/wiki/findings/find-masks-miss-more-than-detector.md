---
id: find-masks-miss-more-than-detector
type: finding
name: Zero-shot masks miss more cells than the detector (miss 0.260 vs 0.150)
summary: 'Against expert markers on 25 frames: masks miss 0.260, false 0.114, jitter 0.239; the pre-only detector 0.150 / 0.165 / 0.215. Restricting to ''ok'' cells raises misses to 0.464.'
status: robust
metrics:
- CV
- HEX
date: '2026-11-09'
source: experiments/rootcause_20261109/out/q1_seg_loose.json
literature:
- target: '[[pachitariu2025cellposesam]]'
  relation: closes
  note: zero-shot masks miss 0.260 vs detector 0.150 on the 25 marker frames; mask-based CV/HEX routes closed
- target: '[[reith2025selfadapt]]'
  relation: suggests
  note: domain adaptation would be the fix; closed by dec-014
- target: '[[marks2025cellsam]]'
  relation: explains
  note: generalist SAM segmenters hold up only near their training distribution
- target: '[[archit2025microsam]]'
  relation: suggests
  note: fine-tuning SAM for microscopy; not pursued after dec-014
- target: '[[kucharski2021cnnwatershed]]'
  relation: closes
  note: zero-shot masks miss 0.260 vs detector 0.150 on the 25 marker frames; mask-based CV/HEX routes closed
- target: '[[schmidt2018stardist]]'
  relation: closes
  note: zero-shot masks miss 0.260 vs detector 0.150 on the 25 marker frames; mask-based CV/HEX routes closed
- target: '[[sierra2023sdm]]'
  relation: closes
  note: zero-shot masks miss 0.260 vs detector 0.150 on the 25 marker frames; mask-based CV/HEX routes closed
- target: '[[viguerasguillen2022feedbackattn]]'
  relation: closes
  note: zero-shot masks miss 0.260 vs detector 0.150 on the 25 marker frames; mask-based CV/HEX routes closed
- target: '[[qu2020pointnuclei]]'
  relation: closes
  note: zero-shot masks miss 0.260 vs detector 0.150 on the 25 marker frames; mask-based CV/HEX routes closed
---
# Zero-shot masks miss more cells than the detector (miss 0.260 vs 0.150)

Against expert markers on 25 frames: masks miss 0.260, false 0.114, jitter 0.239; the pre-only detector 0.150 / 0.165 / 0.215. Restricting to 'ok' cells raises misses to 0.464.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
