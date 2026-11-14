---
id: find-cd-no-cheap-posthoc-lever
type: finding
name: 'No cheap post-hoc CD lever is left: multipliers, clipping, recalibration and blending earlier models all read +0.001 to +0.05'
summary: 'On the frozen candidate''s strict CD OOF every post-hoc lever is worse: global multiplier +0.0046, per-level multiplier +0.0050, clipping +0.0384, per-quintile factors +0.0061, isotonic +0.050, blending submission 1 +0.0024 (+0.0011 with its stretch), blending the Phase I best model +0.0025. The last stage is already a MAPE-optimal log-affine stretch.'
status: robust
metrics:
- CD
date: '2026-11-13'
source: shared notes ledger_20261111.jsonl F-CDANAT-001 (line 92), O-CD-SUB1-CALIB (line 204), O-CD-PHASE1-CALIB (line 205), O-CD-FFT-001 (line 106); experiments/lane_cd_20261108/reads/cd_levers_20261111.json, reads/cd_sub1_calib_20261113.json, reads/cd_phase1_calib_20261113.json
evidence_for:
- '[[dec-018-cd-hypotheses-after-leaderboard]]'
---

# No cheap post-hoc CD lever is left: multipliers, clipping, recalibration and blending earlier models all read +0.001 to +0.05

On the frozen candidate's strict CD OOF every post-hoc lever is worse: global multiplier +0.0046, per-level multiplier +0.0050, clipping +0.0384, per-quintile factors +0.0061, isotonic +0.050, blending submission 1 +0.0024 (+0.0011 with its stretch), blending the Phase I best model +0.0025. The last stage is already a MAPE-optimal log-affine stretch.

## Evidence

[[exp-cd-cheap-posthoc-levers]] (11-11, vs 6.1743 strict), [[exp-cd-sub1-calibration-read]] (11-13, vs 6.1737), [[exp-cd-phase1-calibration-read]] (11-13, folds 2-4). The only reads below zero are inside noise: the Phase I model's sixth member -0.0005 (CI -0.017 to +0.016). The label-free Fourier column gets weight 0 ([[exp-cd-fourier-ring-cd-stack]]).

## Limits

CD only. The Phase I read covers folds 2-4 and uses a development checkpoint for one member. Levers closed earlier (disagreement shift, mask variants, per-source stretch) were not rerun.

## Bears on

Answers D-CDANAT-001 ('is there an untried cheap lever worth more than 0.03 on CD': no) and the PI's calibration question of 11-13. What is left of the CD error is in [[find-cd-p1-error-within-donor-label-scatter]].
