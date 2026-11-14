---
id: dec-018-cd-hypotheses-after-leaderboard
type: decision
name: 'CD after the 11-11 leaderboard: error anatomy, CD-only loss, Fourier ring, native tiles (D-CDANAT-001, D-CDTRAIN-001 to 003, D-CDSIMPLE-001)'
summary: 'With second place at CD 5.975 on the 11-11 leaderboard, the PI put two hypotheses on CD (members under-trained for it; something in the pipeline suppresses it) and said there is surely a simple way to lower it. The coordinator ordered a CPU anatomy of the CD error and four fixed-form arms outside the freeze: a CD-only loss on recipe A at two encoder LRs, a label-free Fourier ring column, and native-tile CD-only training. All were closed by 12:45 EST 11-12; the freeze did not change.'
status: accepted
date: '2026-11-11'
deciders:
- PI
- coordinator
source: shared notes ledger_20261111.jsonl O-LB-001 (line 88), D-CDANAT-001 (line 89), D-CDTRAIN-001 (line 90), D-CDSIMPLE-001 (line 91), D-CDTRAIN-002 (line 104), D-CDTRAIN-003 (line 118), D-SYN-005 (line 131)
decides:
- '[[exp-cd-error-anatomy]]'
- '[[exp-cd-cheap-posthoc-levers]]'
- '[[exp-cd-epoch-probe]]'
- '[[exp-cd-cdonly-recipe-a-elr1e5]]'
- '[[exp-cd-cdonly-recipe-a-elr3e5]]'
- '[[exp-deploy-fourier-ring-cd-alone]]'
- '[[exp-cd-fourier-ring-cd-stack]]'
- '[[exp-bb-native-tile-cd-only]]'
- '[[exp-cd-native-tile-stack-read]]'
- '[[idea-cd-only-loss-recipe-a]]'
- '[[idea-fourier-ring-cd-column]]'
- '[[idea-native-tile-cd-only-training]]'
motivated_by:
- '[[exp-ref-submission1]]'
- '[[exp-cd-from-scratch-search]]'
depends_on:
- '[[dec-016-freeze-candidates-20261111]]'
---

# CD after the 11-11 leaderboard: error anatomy, CD-only loss, Fourier ring, native tiles (D-CDANAT-001, D-CDTRAIN-001 to 003, D-CDSIMPLE-001)

With second place at CD 5.975 on the 11-11 leaderboard, the PI put two hypotheses on CD (members under-trained for it; something in the pipeline suppresses it) and said there is surely a simple way to lower it. The coordinator ordered a CPU anatomy of the CD error and four fixed-form arms outside the freeze: a CD-only loss on recipe A at two encoder LRs, a label-free Fourier ring column, and native-tile CD-only training. All were closed by 12:45 EST 11-12; the freeze did not change.

## Context

Leaderboard from the PI, 11-11 22:40 EST (O-LB-001): 1st V14 8.6920 (CD 6.049, CV 9.928, HEX 10.099), 2nd LKSOM 3 8.7035 (10 Nov; CD 5.975, CV 10.060, HEX 10.075), 3rd LKSOM 2 8.7805, 4th ec blend 8.7816, 5th ours (submission 1) 8.9675 (CD 6.250, CV 10.143, HEX 10.510). The same line carries arithmetic, not a forecast: submission 1's test score plus the strict OOF change gives CD 6.112, CV 10.055, HEX 10.108, mean 8.758 for the candidate. Stack-level suppression of CD was already ruled out (from-scratch search 6.1733 vs 6.1741; every leave-one-group-out is worse).

## Decision

- D-CDANAT-001: CPU anatomy of the CD error of D-CD-001; first question: is there an untried cheap lever worth more than 0.03 on CD.
- D-CDTRAIN-001: a CPU probe (is CD still falling at the last epoch) and recipe A s46 with the CD loss only; read in place of plain recipe A s46 in the CD stack, strict, incumbent in the same run; no strict CD gain on the screen folds closes the arm.
- D-CDTRAIN-002 (supersedes the card plan of D-CDTRAIN-001): two encoder LRs (1e-5, 3e-5) on fold 0; the 1.5x-epochs arm is dropped because the probe shows no fall after epoch 12.
- D-CDTRAIN-003: fold 4 for both arms, so each gets a folds 0+4 strict read; folds 1-3 only if more than 0.01 below the same-run incumbent with both folds lower.
- D-CDSIMPLE-001: (a) a Fourier ring estimate of cell spacing as a label-free CD column (deploy lane, CPU); (b) native-resolution random-tile training with tile-averaged inference, CD loss only (backbone lane). Each is one fixed form, no tuning.

## Consequences

- Anatomy: no untried cheap lever above 0.03 ([[exp-cd-cheap-posthoc-levers]], [[find-cd-no-cheap-posthoc-lever]]); the error sits in source P1 and in the label tails ([[exp-cd-error-anatomy]], [[find-cd-p1-error-within-donor-label-scatter]]).
- Probe: CD is not still falling at the last epoch ([[exp-cd-epoch-probe]]).
- CD-only loss: 1e-5 -0.0030, 3e-5 +0.0181 on folds 0+4 strict; both closed ([[exp-cd-cdonly-recipe-a-elr1e5]], [[exp-cd-cdonly-recipe-a-elr3e5]]).
- Fourier ring: weight 0 in every fold ([[exp-deploy-fourier-ring-cd-alone]], [[exp-cd-fourier-ring-cd-stack]]).
- Native tiles: member +0.255, added +0.0001 ([[exp-bb-native-tile-cd-only]], [[exp-cd-native-tile-stack-read]]).

D-CDTRAIN-003 also parked the synthetic recipe A arm ([[dec-017-synthetic-pretraining-followup]]).

## Revisit when

Not planned: all four arms are closed as tested and the GPUs are down ([[dec-021-gpu-shutdown-closeout-20261113]]). What was never varied on recipe A is listed in [[idea-recipe-a-hyperparameters-never-varied]].
