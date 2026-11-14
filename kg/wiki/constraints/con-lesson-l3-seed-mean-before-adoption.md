---
id: con-lesson-l3-seed-mean-before-adoption
type: constraint
name: 'L3: read a one-seed gain with the seed mean before adoption'
summary: 'Cue: a gain resting on one trained seed. Lesson: order-loss DINOv3 in the CV stack was -0.0460 with seed 46 and -0.0369 with the 3-seed mean; cvw3 in the HEX chain was +0.0027 with seed 46 and -0.0029 with seed 47. Gate: read with the seed mean before adoption; never pick a seed.'
status: active
origin: claude
date: '2026-11-11'
scope: global
source: docs/LANES_20261108.md 'Shared ledger, decision ids, lessons, verifier' (PI approved as rules 2026-11-11; wording and numbers are the coordinator's); docs/FRAMEWORK_REVISION_20261111.md section 3; shared notes ledger_20261111.jsonl L3
---

# L3: read a one-seed gain with the seed mean before adoption

## Rule

Cue: a gain resting on one trained seed. Lesson: order-loss DINOv3 in the CV stack was -0.0460 with seed 46 and -0.0369 with the 3-seed mean; cvw3 in the HEX chain was +0.0027 with seed 46 and -0.0029 with seed 47. Gate: read with the seed mean before adoption; never pick a seed.

## Notes

A pre-launch gate: before a new GPU direction, answer the cue in one line in the first RESULTS.tsv row. The gates decide how a number is read and quoted; they add no CI or minimum-effect threshold ([[con-no-statistical-gates]]). Adopted by [[dec-015-framework-revision-20261111]].
