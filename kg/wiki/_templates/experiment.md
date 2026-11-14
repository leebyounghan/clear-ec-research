---
id: exp-<lane>-<short-name>         # lane prefix: cd, cv, hex, bb (backbone), deploy, gpu0..3 (old lanes 11-06), hist (before lanes), ref (baselines, verifier)
type: experiment
name: ""                            # what was tried, in a few words
summary: ""                         # one or two sentences: what, result, verdict
lane: cd                            # cd | cv | hex | backbone | deploy | gpu0-1106 | gpu1-1106 | gpu2-1106 | gpu3-1106 | pre-lanes | coordinator
date: 2026-11-08                    # as written in the source (some lanes write UTC dates)
status: dropped                     # kept | dropped | running | closed | superseded | reference | measurement
direction: ""                       # the direction this try belongs to; tries of one direction share the exact string
metrics: [CD]                       # metrics reported
primary_metric: CD                  # metric the delta refers to (CD | CV | HEX | mean)
folds: "0,4"                        # MANDATORY: "0,4" | "0-4" | "0" | "25 expert frames" ...
baseline: ""                        # MANDATORY: what the delta is against, with its value on the same rows
baseline_ref: ""                    # "[[exp-ref-...]]" when the baseline is a node
value: ""                           # as written, e.g. "CD 6.187"
delta: ""                           # as written, e.g. "-0.017 vs stack 6.204; -0.077 vs best pipeline"
delta_num:                          # number for sorting, against `baseline` (negative = better); empty if unclear
ambiguous: false                    # true when a number, baseline or fold set is unclear in the source
ambiguous_note: ""
source: ""                          # file and row, e.g. "experiments/lane_cd_20261108/RESULTS.tsv line 51 (snapshot 2026-11-09 02:00 UTC)"
uses_method: []                     # MANDATORY in practice: "[[method-...]]"
uses_dataset: []
stacked_on: []                      # kept step(s) it was built on
superseded_by: []
supports_finding: []
contradicts_finding: []
motivated_by: []
---

# {name}

## What was tried

## Result
Numbers as written in the source, with the baseline and folds in the same sentence.

## Versions
Other versions in the same direction and how they read.

## Notes
