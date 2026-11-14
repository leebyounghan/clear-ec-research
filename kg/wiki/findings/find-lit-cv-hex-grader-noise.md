---
id: find-lit-cv-hex-grader-noise
type: finding
name: 'Literature: inter-grader ICC for CV and HEX is <= 0.265 (eye bank) and < 0.65 (clinic); label noise exceeds the ~150-cell sampling floor'
summary: Eye-bank graders on Konan CellChek D (vidalvillegas2024eyebank) agree on CV and hexagonality with ICC <= 0.265 (ECD 0.78-0.82); clinic examiners on two devices (desanctis2006twodevices) reach ICC < 0.65 for CV and HEX (> 0.85 for ECD). Grader and device behaviour add to the ~6.5 % / 7.5 % sampling SE, so the sampling SE is not the label-noise floor.
status: tentative
metrics:
- CV
- HEX
date: '2026-11-09'
source: paper notes vidalvillegas2024eyebank (key_numbers, from abstract) and desanctis2006twodevices (key_numbers, from abstract)
contradicts:
- target: '[[find-sampling-se-150-cells]]'
  note: 'assumption contradicted: that the ~150-cell sampling SE (CV ~6.5 %, HEX ~7.5 %) is the whole label-noise floor; grader/device noise adds to it. The SE numbers themselves stand.'
evidence_for:
- target: '[[find-residual-at-oracle-floor-cv-hex]]'
  note: 'consistent: 78 % (CV) / 88 % (HEX) of our residual is device behaviour plus label noise beyond sampling'
qualifies:
- target: '[[find-hex-near-ceiling]]'
  note: if label noise exceeds sampling, the HEX ceiling sits even closer than the sampling SE suggests
literature:
- target: '[[vidalvillegas2024eyebank]]'
  relation: source
  note: inter-grader ICC CV and HEX <= 0.265; ECD 0.821 (centre) / 0.784 (flex-centre)
- target: '[[desanctis2006twodevices]]'
  relation: source
  note: inter-examiner ICC < 0.65 for CV and HEX; > 0.85 for ECD
- target: '[[scarpa2016morphometry]]'
  relation: explains
  note: 'expert vs expert on the same cells differ by only CV 2.9 %, HEX 2.6 %: low noise when the cells are fixed; the large ICC gap comes from choosing different cells/areas'
---
# Literature: inter-grader ICC for CV and HEX is <= 0.265 (eye bank) and < 0.65 (clinic); label noise exceeds the ~150-cell sampling floor

Eye-bank graders on Konan CellChek D (vidalvillegas2024eyebank) agree on CV and hexagonality with ICC <= 0.265 (ECD 0.78-0.82); clinic examiners on two devices (desanctis2006twodevices) reach ICC < 0.65 for CV and HEX (> 0.85 for ECD). Grader and device behaviour add to the ~6.5 % / 7.5 % sampling SE, so the sampling SE is not the label-noise floor.

## Evidence

- vidalvillegas2024eyebank (abstract): inter-grader ICC for CV and hexagonality <= 0.265; ECD centre 0.821, flex-centre 0.784; Konan CellChek D in an eye bank (closest analogue of the CLEAR-EC label process).
- desanctis2006twodevices (abstract): inter-examiner ICC > 0.85 for average cell size and ECD, < 0.65 for CV and hexagonality; Topcon vs Konan CV/HEX r < 0.20.

## Limits

Both numbers are from abstracts; populations and spreads differ from CLEAR-EC. An ICC of 0.265 cannot hold as-is for our CV labels: the best pipeline reaches CV r 0.686 with the label ([[find-cv-error-in-tails]]), and a predictor's correlation with a one-grader label is bounded by sqrt(label reliability), so our CV label reliability is at least 0.686^2 = 0.47 (arithmetic). The direction (grader noise > sampling noise) holds; the size does not transfer.

## Bears on

Lowers the expected headroom on CV and HEX from cell reading ([[con-bottleneck-first]]); argues for noise-aware heads and decoding ([[idea-distribution-head-cv-hex]]) over tail chasing ([[idea-imbalanced-regression-cv-tails]]).
