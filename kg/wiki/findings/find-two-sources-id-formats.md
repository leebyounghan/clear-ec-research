---
id: find-two-sources-id-formats
type: finding
name: Two ID formats mark two sources; CD error 7.11 (old) vs 5.50 (new)
summary: An image classifier tells the two sources apart (AUC 0.998); CD error differs by source.
status: robust
metrics:
- CD
date: '2026-11-08'
source: 'docs/BRIEF.md ''Measured facts: Labels'''
literature:
- target: '[[adachi2025ssa]]'
  relation: suggests
  note: check BN-adapt vs SSA on the old/new source split before any test-time adaptation
---
# Two ID formats mark two sources; CD error 7.11 (old) vs 5.50 (new)

An image classifier tells the two sources apart (AUC 0.998); CD error differs by source.

## Limits and notes

Any model or correction keyed on source is ruled out (con-no-source-id-shortcuts).

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.

## Update 2026-11-13

On the frozen candidate's strict CD OOF the two sources read P1 7.04 and P2 5.29, P1 carrying 57.5 % of the error; the gap is mostly label scatter within a donor ([[exp-cd-error-anatomy]], [[find-cd-p1-error-within-donor-label-scatter]]).
