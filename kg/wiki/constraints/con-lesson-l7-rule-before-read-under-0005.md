---
id: con-lesson-l7-rule-before-read-under-0005
type: constraint
name: 'L7: for differences under 0.005, fix the rule before the read; no cascades'
summary: 'Cue: adopting or dropping on a difference under 0.005. Lesson: seed noise is about 0.005 and leave-one-out cascades move 0.001. Gate: fix the rule before the read; no cascades.'
status: active
origin: claude
date: '2026-11-11'
scope: global
source: docs/LANES_20261108.md 'Shared ledger, decision ids, lessons, verifier' (PI approved as rules 2026-11-11; wording and numbers are the coordinator's); docs/FRAMEWORK_REVISION_20261111.md section 3; shared notes ledger_20261111.jsonl L7
---

# L7: for differences under 0.005, fix the rule before the read; no cascades

## Rule

Cue: adopting or dropping on a difference under 0.005. Lesson: seed noise is about 0.005 and leave-one-out cascades move 0.001. Gate: fix the rule before the read; no cascades.

## Notes

A pre-launch gate: before a new GPU direction, answer the cue in one line in the first RESULTS.tsv row. The gates decide how a number is read and quoted; they add no CI or minimum-effect threshold ([[con-no-statistical-gates]]). Adopted by [[dec-015-framework-revision-20261111]].
