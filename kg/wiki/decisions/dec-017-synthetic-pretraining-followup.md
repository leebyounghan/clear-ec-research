---
id: dec-017-synthetic-pretraining-followup
type: decision
name: 'Synthetic pretraining: one bounded follow-up on ViT-L and recipe A (D-SYN-002, D-SYN-003)'
summary: 'Closed 2026-11-12. After D-SYN-001 (synthft3 DINOv3-B not adopted in any stack), one bounded follow-up applied synthetic pretraining to the members that carry the stacks: ViT-L s46 (D-SYN-002, extended to five folds by D-SYN-004) and recipe A s46 (D-SYN-003, extended by D-SYN-005). Neither holds in a stack on five folds strict: the PI kept the synthetic ViT-L out of the CD stack (D-SYN-006) and the coordinator closed synthetic recipe A (D-SYN-008). The freeze is unchanged.'
status: accepted
date: '2026-11-11'
deciders:
- coordinator
- PI
source: shared notes ledger_20261111.jsonl D-SYN-001 (line 13), D-SYN-002 (line 50), D-SYN-003 (line 66), D-SYN-004 (line 81), D-CDTRAIN-003 (line 118), D-SYN-005 (line 131), D-SYN-006 (line 136), D-SYN-007 (line 164), D-SYN-008 (line 181); docs/FINAL_SUMMARY_20261113.md section 3.2
decides:
- '[[idea-synthetic-specular-pretraining]]'
- '[[exp-bb-synth-vitl]]'
- '[[exp-bb-synth-recipe-a]]'
- '[[exp-cd-synth-vitl-stack]]'
- '[[exp-cv-synth-vitl-stack]]'
- '[[exp-cd-synth-recipe-a-stack]]'
- '[[exp-cv-synth-recipe-a-swap]]'
- '[[exp-hex-synth-recipe-a-swap]]'
- '[[exp-bb-synth-recipe-a-seed47-fold0]]'
motivated_by:
- '[[exp-bb-synthft3-dinov3]]'
- '[[exp-bb-synth-vitl]]'
- '[[exp-bb-synth-recipe-a]]'
depends_on:
- '[[dec-016-freeze-candidates-20261111]]'
---

# Synthetic pretraining: one bounded follow-up on ViT-L and recipe A (D-SYN-002, D-SYN-003)

Closed 2026-11-12. After D-SYN-001 (synthft3 DINOv3-B not adopted in any stack), one bounded follow-up applied synthetic pretraining to the members that carry the stacks: ViT-L s46 (D-SYN-002, extended to five folds by D-SYN-004) and recipe A s46 (D-SYN-003, extended by D-SYN-005). Neither holds in a stack on five folds strict: the PI kept the synthetic ViT-L out of the CD stack (D-SYN-006) and the coordinator closed synthetic recipe A (D-SYN-008). The freeze is unchanged.

## Decision

Kill rule: no strict stack gain on folds 0+4 in CD or CV closes synthetic pretraining for stacks. A positive read goes to the PI before any five-fold or full-data run.

## What followed (2026-11-11 22:10 EST to 2026-11-13)

| Id | Who, when (EST) | Decision | Value recorded |
|---|---|---|---|
| D-SYN-002 | coordinator, 11-11 15:00 | ViT-L s46, folds 0 and 4, read in place of the plain member, strict | CV screen -0.0154 (10.0297 vs 10.0451); CD screen -0.0012 (6.1576 vs 6.1588) |
| D-SYN-003 | coordinator, 11-11 15:30 | recipe A s46, folds 0 and 4 | CD screen -0.0059 (6.1528 vs 6.1588); CV +0.0094 folds 0+4, +0.0041 five folds; HEX +0.0017 five folds |
| D-SYN-004 | coordinator, 11-11 22:10 | synthetic ViT-L folds 1-3 after the positive CV screen; candidate for the PI only if strict five-fold CV is below 10.1160 or the CD stack below 6.1743 | CV 10.1237 (+0.0078): not a candidate; CD 6.1712 (-0.0031): a CD-side candidate by the rule, under seed noise, folds mixed |
| D-CDTRAIN-003 | coordinator, 11-12 02:45 | synthetic recipe A not extended to further folds for now (parked; cards go to the CD-only arms) | - |
| D-SYN-005 | coordinator, 11-12 12:00 | idle GPUs: synthetic recipe A folds 1-3 and the full-data synthetic ViT-L build, as option-keeping work | recipe A five folds: member -0.0063; CD strict -0.0017 |
| D-SYN-006 | PI, 11-12 13:45 | do not put the synthetic ViT-L into the CD stack; D-CD-001 stands; the f99 build is stopped and its partial output archived, not deleted | build stopped at epoch 9 of 12 |
| D-SYN-007 | coordinator's own call, 11-12 18:10 | synthetic recipe A fold 0 with seed 47 on the idle GPU 2, for the seed mean L3 requires; information only | fold-0 seed mean -0.022 (CI -0.057 to +0.016) |
| D-SYN-008 | coordinator, 11-12 22:40 | synthetic-pretrained recipe A is closed | CD strict 6.1726 vs 6.1743 (-0.0017, 95 % CI -0.012 to +0.008), per-fold signs mixed; member level with plain |

The entries that the 11-11 ingest left as 'No value recorded yet' (D-SYN-002, D-SYN-003) are filled in above; the experiment nodes hold the full reads: [[exp-bb-synth-vitl]], [[exp-cv-synth-vitl-stack]], [[exp-cd-synth-vitl-stack]], [[exp-bb-synth-recipe-a]], [[exp-cd-synth-recipe-a-stack]], [[exp-cv-synth-recipe-a-swap]], [[exp-hex-synth-recipe-a-swap]], [[exp-bb-synth-recipe-a-seed47-fold0]].

## Consequences

All three synthetic routes are closed for all three metrics ([[find-synthetic-pretraining-gains-vanish-in-stacks]]); nothing changed in the freeze or the packaging. The kill rule of D-SYN-002 (no strict stack gain on folds 0+4 closes it) did not fire on the screens, which is why five folds were run; the five-fold reads closed it (L2).

## Revisit when

Not planned. Never run: the synthetic recipe A seed mean on folds 1-4 and any full-data synthetic build ([[idea-synthetic-recipe-a-seed-mean-and-full-data]]).
