---
id: find-lit-cv-compression-expected
type: finding
name: 'Literature: a prediction spread of 66 % at r 0.686 is what a calibrated regressor produces'
summary: 'For a predictor of E[y|x], the prediction/label SD ratio equals r (delange2022mindgap, paper-note derivation): our CV predictions span 66 % of the label spread at r 0.686, as expected. Compression is the signature of an imperfect regressor, not a fixable bias; label-free de-shrinking inflates variance, and any de-shrink must be fitted on OOF MAPE.'
status: tentative
metrics:
- CV
date: '2026-11-09'
source: 'paper note delange2022mindgap (suggests_try, relevance_note; key_numbers: label-free slope correction inflates variance)'
contradicts:
- target: '[[idea-imbalanced-regression-cv-tails]]'
  note: 'assumption contradicted: that the outer-decile CV error (a third of the error) is an imbalance defect to fix; binned by true label the outer deciles always look bad for any regressor at r 0.686'
qualifies:
- target: '[[find-cv-error-in-tails]]'
  note: the 66 % spread and the tail error are the expected shape at r 0.686, not evidence of a tail-specific bias
evidence_for:
- target: '[[find-cv-hex-worst-rows-label-scatter]]'
  note: 'consistent: worst rows look like label scatter'
literature:
- target: '[[delange2022mindgap]]'
  relation: source
  note: age-bias correction with the true label inflates metrics; label-free slope correction inflates variance
---
# Literature: a prediction spread of 66 % at r 0.686 is what a calibrated regressor produces

For a predictor of E[y|x], the prediction/label SD ratio equals r (delange2022mindgap, paper-note derivation): our CV predictions span 66 % of the label spread at r 0.686, as expected. Compression is the signature of an imperfect regressor, not a fixable bias; label-free de-shrinking inflates variance, and any de-shrink must be fitted on OOF MAPE.

## Evidence

- delange2022mindgap: corrections that use the true target make even shuffled-label models look accurate; the label-free slope correction (divide by slope) inflates variance.
- [[find-cv-error-in-tails]]: best pipeline CV r 0.686, predictions span 66 % of the label spread, outer deciles hold a third of the error, bias flat by prediction decile.
- SD ratio = r for a calibrated predictor (Var E[y|x] = r^2 Var y): 0.686 vs observed 0.66 (paper-note derivation; arithmetic).

## Limits

A literature argument plus arithmetic; it does not say a better model cannot raise r. The per-metric log-affine stretch fitted on OOF ([[method-log-affine-stretch]]) is the endorsed form of de-shrinking and is already in the pipeline.

## Bears on

Argues against tail re-weighting ([[idea-imbalanced-regression-cv-tails]]) and against any stretch not fitted on OOF MAPE.
