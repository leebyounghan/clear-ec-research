---
id: dec-015-framework-revision-20261111
type: decision
name: 'Framework revision: shared ledger, decision ids, lessons L1-L7 as a gate, verifier, probe first (D-FW-001)'
summary: 'The PI asked for the framework to be revised against arXiv:2611.12552 (SciMARS) and approved adding the result to the charter as rules: an append-only shared ledger, versioned decision ids, lessons L1-L7 as a pre-launch gate, a verifier lane (the CD lane) and probe-first.'
status: accepted
date: '2026-11-11'
deciders:
- PI
- coordinator
source: docs/FRAMEWORK_REVISION_20261111.md; docs/LANES_20261108.md 'Shared ledger, decision ids, lessons, verifier'; shared notes ledger_20261111.jsonl D-FW-001
decides:
- '[[con-lesson-l1-member-alone-vs-stack]]'
- '[[con-lesson-l2-screen-folds-not-a-gain]]'
- '[[con-lesson-l3-seed-mean-before-adoption]]'
- '[[con-lesson-l4-quote-strict-value]]'
- '[[con-lesson-l5-deploy-equals-evaluated]]'
- '[[con-lesson-l6-like-for-like]]'
- '[[con-lesson-l7-rule-before-read-under-0005]]'
literature:
- target: '[[wu2026scimars]]'
  relation: source-of-method
  note: shared typed append-only evidence graph, one-change-at-a-time edits, probes with matched comparison, reflection lessons; the paper's own results are small benchmark replays and do not measure what the changes are worth here
---

# Framework revision: shared ledger, decision ids, lessons L1-L7 as a gate, verifier, probe first (D-FW-001)

The PI asked for the framework to be revised against arXiv:2611.12552 (SciMARS) and approved adding the result to the charter as rules: an append-only shared ledger, versioned decision ids, lessons L1-L7 as a pre-launch gate, a verifier lane (the CD lane) and probe-first.

## Context

Between 11-10 afternoon and 11-11 midday recorded findings were not acted on, constraints known to one lane did not reach the others, and the HEX freeze candidate changed chain -> ridge -> chain -> ridge (docs/FRAMEWORK_REVISION_20261111.md).

## Decision

Ledger, decision ids, lessons as a gate, verifier, probe first. Not taken from the paper: automatic editing of lane prompts or roles, per-round re-selection of the team.

## Consequences

The coordinator ingests the ledger into this KG; lanes still do not edit it.
