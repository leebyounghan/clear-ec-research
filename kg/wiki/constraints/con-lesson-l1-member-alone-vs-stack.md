---
id: con-lesson-l1-member-alone-vs-stack
type: constraint
name: 'L1: a member that is better alone does not predict stack weight'
summary: 'Cue: a new image-to-label member or a variant of one. Lesson: a member better alone by 0.01 to 0.07 on the mean got stack weight 0 ten times. Gate: say what information the stack lacks that this adds; stack read on folds 0+4 before any further fold. Reconsider when residual correlation with the stack is below 0.9.'
status: active
origin: claude
date: '2026-11-11'
scope: global
source: docs/LANES_20261108.md 'Shared ledger, decision ids, lessons, verifier' (PI approved as rules 2026-11-11; wording and numbers are the coordinator's); docs/FRAMEWORK_REVISION_20261111.md section 3; shared notes ledger_20261111.jsonl L1
constrains:
- '[[method-nested-blend]]'
---

# L1: a member that is better alone does not predict stack weight

## Rule

Cue: a new image-to-label member or a variant of one. Lesson: a member better alone by 0.01 to 0.07 on the mean got stack weight 0 ten times. Gate: say what information the stack lacks that this adds; stack read on folds 0+4 before any further fold. Reconsider when residual correlation with the stack is below 0.9.

## Notes

A pre-launch gate: before a new GPU direction, answer the cue in one line in the first RESULTS.tsv row. The gates decide how a number is read and quoted; they add no CI or minimum-effect threshold ([[con-no-statistical-gates]]). Adopted by [[dec-015-framework-revision-20261111]].
