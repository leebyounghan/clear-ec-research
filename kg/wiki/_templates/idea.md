---
id: idea-short-name
type: idea
name: ""
summary: ""                         # the hypothesis in one sentence
status: open                        # open | never-tried | running | partially-tested | closed | dropped
origin: claude                      # user (PI) | claude | literature
aims_at: [CV]                       # CD | CV | HEX | all
owner_lane: cv                      # lane that would own it: cd | cv | hex | backbone | coordinator
cost: ""                            # estimate as text, e.g. "~3 CPU-h (25-frame fit) + ~1 CPU-h stack read"
cost_hours:                         # number, for sorting
cost_unit: CPU-h                    # CPU-h | GPU-h
potential_metric: CV                # metric whose gap it targets (CD | CV | HEX)
potential_gain:                     # ESTIMATE of the OOF decrease it could bring on that metric (metric units); empty = unknown
potential_basis: ""                 # why that estimate (closest try's delta, variance share, paper effect size); label it an estimate
source: ""                          # where the idea is listed (LANES, PLAN row, audit, lane report)
novelty_check: ""                   # dated: what was searched in the KG and on the web, and what came back
motivated_by: []                    # find-... / exp-... / paper ids / con-...
builds_on: []                       # method-... / paper ids / ds-...
tested_by: []                       # - {target: "[[exp-...]]", closeness: same|close|related, note: "what differs"}
depends_on: []
---

# {name}

## Hypothesis
## Closest tries in the KG
## Literature
## Minimal test
Folds, baseline, success criterion (lower OOF on folds 0 and 4 against the best pipeline), time box.
## Risks
