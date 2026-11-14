---
id: find-cd-full-extra-weights-nothing-over-t1
type: finding
name: Under strict nesting the full candidate's extra CD weights buy nothing over CD t1 (6.1743 vs 6.1737)
summary: 'CD t1 drops the recipe A e8-19 snapshot and the ViT-L e8-11 single views (about 9 GB of CD-only weights and 32 forwards per image) and reads 6.1737 against the full stack''s 6.1743: -0.0006 (CI -0.007 to +0.006), strict five folds, incumbent recomputed in the same batch. The snapshot''s earlier -0.004 was an as-computed gain.'
status: robust
metrics:
- CD
date: '2026-11-13'
source: shared notes ledger_20261111.jsonl O-CD-TIERS (line 146), D-PKG-003 (line 159); docs/lane_summaries_20261113/final_summary_cd_20261113.md sections 4a and 6; docs/FINAL_SUMMARY_20261113.md section 5
qualifies:
- '[[find-checkpoint-averaging-expectation]]'
evidence_for:
- '[[dec-019-package-tiers-20261112]]'
---

# Under strict nesting the full candidate's extra CD weights buy nothing over CD t1 (6.1743 vs 6.1737)

CD t1 drops the recipe A e8-19 snapshot and the ViT-L e8-11 single views (about 9 GB of CD-only weights and 32 forwards per image) and reads 6.1737 against the full stack's 6.1743: -0.0006 (CI -0.007 to +0.006), strict five folds, incumbent recomputed in the same batch. The snapshot's earlier -0.004 was an as-computed gain.

## Evidence

[[exp-cd-deploy-tiers]]: per fold +0.002 / +0.011 / -0.002 / -0.016 / +0.002. CD-only packaged weights: full candidate ~11.0 GB (recipe A e8-19 4.1, ViT-L e8-11 4.9, ViT-L final 1.2, DINOv3 s47 0.86) against 2.07 GB for t1. The CD lane's table: 'Recipe A snapshot (1/2 e20 TTA + 1/2 e8-19 single): -0.004 as computed (11-08); zero under strict (t1 = full)' ([[exp-cd-flip-tta-snapshot]]).

## Limits

One strict read; a difference of 0.0006 is far inside seed noise (0.005, L7) in either direction, so the statement is 'no measurable loss', not 't1 is better'.

## Bears on

Tier p and tier t1 carry CD t1 ([[dec-019-package-tiers-20261112]]). An SWA average in place of epoch snapshots is moot after t1 ([[idea-ema-swa]]). It is another case of an as-computed gain that strict nesting removes ([[con-lesson-l4-quote-strict-value]]).
