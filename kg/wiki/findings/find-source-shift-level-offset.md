---
id: find-source-shift-level-offset
type: finding
name: Across sources all three metrics lose, 85-90 % of it a flat level offset (F-SHIFT-001)
summary: 'Leave-one-source-out penalties: CD +3.4 / +3.7, CV +6.0 / +6.5, HEX +3.5 / +0.8 MAPE points; 85-90 % of the loss is a flat level offset; the mixed-source candidate has an offset of about 0 by source. Shading correction raises the offset (0.0825 -> 0.0959). Test-time alignment was not started.'
status: robust
metrics:
- CD
- CV
- HEX
date: '2026-11-11'
source: shared notes ledger_20261111.jsonl F-SHIFT-001; lane_backbone_20261108/reads/shift_step1_raw.json; lane_backbone_notes.md '2026-11-11 05:15 EST'; O-BB-SYNFT3-5F-CORR (ledger line 208); docs/lane_summaries_20261113/final_summary_backbone_20261113.md sections 4b and 6
qualifies:
- target: '[[find-hex-test-gap]]'
  note: 'no HEX-specific shift penalty: the shift is a source level offset'
evidence_against:
- target: '[[idea-shading-correction]]'
  note: shading correction raised the source offset 0.0825 -> 0.0959
motivated_by:
- '[[find-two-sources-id-formats]]'
---

# Across sources all three metrics lose, 85-90 % of it a flat level offset (F-SHIFT-001)

Leave-one-source-out penalties: CD +3.4 / +3.7, CV +6.0 / +6.5, HEX +3.5 / +0.8 MAPE points; 85-90 % of the loss is a flat level offset; the mixed-source candidate has an offset of about 0 by source. Shading correction raises the offset (0.0825 -> 0.0959). Test-time alignment was not started.

## Evidence

- [[exp-bb-leave-one-source-out]], [[exp-bb-shading-correction]].

## Limits

Two sources (the two ID formats); which penalty belongs to which held-out source is not stated in the ledger line. There is no HEX-specific shift penalty.

## Bears on

[[idea-shading-correction]] (closed), [[idea-appearance-aug-hex-gap]], [[idea-test-time-adaptation-regression]], [[idea-leave-one-source-out-harness]].

## Update 2026-11-13

Correction on record (O-BB-SYNFT3-5F-CORR, 11-13): the label-decile bias read under this finding (`lane_backbone_20261108/reads/shift_decile_bias_20261111.json`) used the 11-10 integration-rehearsal OOF as the 'candidate', not the frozen candidate; the statement 'the mixed-source candidate has an offset of about 0 by source' therefore refers to the rehearsal OOF. The leave-one-source-out penalties are unaffected. The backbone lane's summary names the sources P1 / P2 for the pairs (CD +3.4 / +3.7, CV +6.0 / +6.5, HEX +3.5 / +0.8) and states that there is no HEX-specific penalty. Test-time subspace alignment was not run.
