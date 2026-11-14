---
id: find-model-reads-grain
type: finding
name: 'The model reads image grain: blur without grain moves CV +6 %, with grain refilled -1.9 %'
summary: Inside the real range single frames move ~1 % and MAPE on 240 frames rises 0.48 under measured transfer.
status: robust
metrics:
- CV
- CD
- HEX
date: '2026-11-08'
source: experiments/lane_gpu0_20261106/REPORT.md 'Response of a trained member'
literature:
- target: '[[stringer2025cellpose3]]'
  relation: explains
  note: restoration targets noise, blur and undersampling, not vignetting
---
# The model reads image grain: blur without grain moves CV +6 %, with grain refilled -1.9 %

Inside the real range single frames move ~1 % and MAPE on 240 frames rises 0.48 under measured transfer.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
