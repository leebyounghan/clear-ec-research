---
id: find-sampling-se-150-cells
type: finding
name: About 150 cells give a standard error near 6.5 % on CV, 7.5 % on HEX, 3 % on CD
summary: If the marked cells were independent draws, the label itself carries this sampling noise.
status: robust
metrics:
- CV
- HEX
- CD
date: '2026-11-08'
source: docs/BRIEF.md 'Sampling'
evidence_for:
- '[[find-residual-at-oracle-floor-cv-hex]]'
literature:
- target: '[[doughty2018regional]]'
  relation: corroborates
  note: with <= 50 cells COV can differ by +-10 %; 75-100 cells recommended
- target: '[[doughty2020cvcells]]'
  relation: corroborates
  note: COV reliability +-1.0 to +-5.0 % with 90 cells, worse at 75; grows with CV
- target: '[[mccarey2008review]]'
  relation: corroborates
  note: 'repeat analysis of one pattern: CV 41 +- 1 (78 cells), 48 +- 1 (147 cells)'
- target: '[[marrugo2026review]]'
  relation: corroborates
  note: 'small analysed areas unstable: four acquisitions of one region 1100-1426 cells/mm2'
- target: '[[scarpa2016morphometry]]'
  relation: explains
  note: 'expert vs expert on the same cells: CV 2.9 %, HEX 2.6 %, ECD 0.5 %'
---
# About 150 cells give a standard error near 6.5 % on CV, 7.5 % on HEX, 3 % on CD

If the marked cells were independent draws, the label itself carries this sampling noise.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.
