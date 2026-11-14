---
id: method-log-affine-stretch
type: method
name: Log-affine stretch
summary: 'Per-metric affine map in log space fitted on OOF (nested: fitted on the other folds), correcting the shrinkage of predictions toward the mean.'
literature:
- target: '[[delange2022mindgap]]'
  relation: corroborates
  note: de-shrinking must be fitted on OOF error, not label-free
- target: '[[tofallis2015logratio]]'
  relation: explains
  note: log-space fitting targets the geometric mean; MAPE optimum lies below it
---
# Log-affine stretch

Per-metric affine map in log space fitted on OOF (nested: fitted on the other folds), correcting the shrinkage of predictions toward the mean.

## Where it is used

Experiments link here with `uses_method`; see backlinks or `kg_query.py node method-log-affine-stretch`.
