---
id: find-impossible-labels
type: finding
name: About 25 rows carry labels the image contradicts (worth -0.18, unpredictable)
summary: 'E.g. B 2020 serial 644-651: six frames with CD labels 372-926 where the model reads 1560-3010; two HEX-0 rows with CV 0.03 and 0.11 hold 0.19 on CV.'
status: robust
metrics:
- CD
- CV
date: '2026-11-08'
source: 'docs/LANES_20261108.md ''Facts per lane: CD''; experiments/lane_gpu0_20261106/REPORT.md ''Extreme rows''; memory/clear-ec-error-case-analysis.md; shared notes ledger_20261111.jsonl O-CDANAT-001 (line 93)'
literature:
- target: '[[kim2024confrag]]'
  relation: suggests
  note: down-weight frames with large, consistent OOF residuals (CD only)
---
# About 25 rows carry labels the image contradicts (worth -0.18, unpredictable)

E.g. B 2020 serial 644-651: six frames with CD labels 372-926 where the model reads 1560-3010; two HEX-0 rows with CV 0.03 and 0.11 hold 0.19 on CV.

## Limits and notes

With APE clipped at 50 the OOF mean is 8.568 vs 8.739. These rows land on test by chance.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.

## Update 2026-11-13

On the frozen candidate's strict CD OOF (6.1743) the worst 0.1 % of rows (9 rows) carry 4.0 % of the total APE; 6 of them are the serials 2020-0644..0651, worth 0.187 CD MAPE; only 6 rows have labels more than 2x off the calibrated mask CD ([[exp-cd-error-anatomy]]). Unpredictable; no action taken.
