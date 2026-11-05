# Phase II decision plan (written 2026-10-30, before any Phase II submission)

> Submission-2 plan superseded on 2026-11-03: see [STATE_20261103.md](STATE_20261103.md).

## Revision 2026-10-30 late (supersedes Sections 2-4 where they differ)

Corrections after PI review:
- Failed evaluations do not consume submission quota (Phase I: limit 3, four entries incl. one failure).
- Submitting early is not free: our 1000-image score becomes public before competitors pick their second entry.
- The label-noise "floor" in Section 5 was overstated: the metric is mean absolute % error, so a binomial
  ROI model with ~125-150 cells gives a HEX/CV floor of ~5.5 % MAPE, against our ~10 %. Unknown, must be measured.
- Bands 2a/2b and the robustness audit addressed a low-probability case (no shift signal in Phase I) and
  are dropped.

Current plan:
1. Submission 1 = hybrid B, after a platform Try-out on T4, around 2026-11-08 (keeps ~1 week to repair).
2. Submission 2 = B + validated increments (hex6 per fixed rule, SSL-seed members if gated). Change course
   only if submission 1 fails or is absurd; do not react to its score otherwise.
3. Add a never-crash fallback guard: hidden-test failures cannot be seen or reproduced by us.
4. First research priority: measure the label-noise floor from train_green ROI points (CPU, hours).
   Floor near 9 % -> stop modelling, run submissions only. Floor near 6 % -> spend the remaining time on
   what causes the gap, not on increments.

Why small gains still matter (proposal of 2026-10-30, 20:30 UTC): invisible on the board is not worthless.
If the score difference to a rival has SD ~0.2 on 1000 images, P(we beat them) = Phi(-gap/0.2); near gap 0 each
0.05 pp of expected gain adds ~10 points of win probability. B's -0.09 vs the Phase-1 six is worth ~15-18 points.
Best-of-two arithmetic: B + validated increments (expected -0.04, paired SD ~0.05) improves E[min] by ~0.046;
a deliberately different but worse candidate (+0.07, SD 0.1) only by ~0.014.

Timeline (proposal):
| Date | Action |
|---|---|
| 10/31-11/01 | label-noise floor (CPU); fallback guard, rebuild, n60 + container re-test; eval harness + residual report |
| 11/01-11/02 | PI uploads image + model to the algorithm; Try-out on 3-5 training images (T4 time, outputs, zero fallbacks) |
| ~11/01 | green confirmation verdict -> hex6 composition fixed by the pre-set rule |
| ~11/02 | SSL-seed gate (>= 0.03 on 2 folds); pass -> full-data SSL + 3 regressions 11/03-11/06 |
| 11/03-11/06 | post-hoc combination ideas on OOF (dev folds 0-3, one confirmation on fold 4) |
| 11/07 | submission 1 = hybrid B |
| 11/09 | submission-2 package frozen; 11/10 Try-out |
| 11/11-11/12 | submission 2 = B + passed increments |
| 11/13-11/15 | buffer (deadline 11/15 06:00 Toronto) |

Not doing: waiting for rivals as a condition (their scores cannot change submission 2); robustness audit and
bands; any new architecture that needs a five-fold retrain (4-5 days, collides with the 11/09 freeze).

Effect sizes that justify not over-deciding: submission timing ~0; submission-2 composition ~0.03;
between-team noise on 1000 images ~0.2; Phase I gap to 1st 0.08.

Purpose: fix, before we see any Phase II score, (1) what submission 1 is for, (2) how its single
observable number will be read, (3) which pre-committed action follows in each band, and (4) what must
be prepared now so that the action in each band is executable in the time left. Deadline 2026-11-15
06:00 America/Toronto; two submissions, best counts; the leaderboard shows the score immediately.

## 1. What the platform will tell us

| Observable | Phase I evidence | Use |
|---|---|---|
| Score S (one scalar) | Phase I pages show only `Score`; metrics.json with per-metric means was exposed only in the debug phase | assume S only |
| Pass/fail per submission | failed evaluation = no score (our 2026-08-07 14:18 submission) | insurance branch |
| Other teams' Phase II scores vs their Phase I scores | public leaderboards | free covariate for "is the test harder for everyone?" |
| Runtime | Phase-1 container: exec 44 s on T4 (37.8 s inference), total 279 s incl. provisioning | Try-out before each submission |

Noise model (per-image absolute-percent-error SD ~10 pp, 1000 images):
- level SE ~0.3 pp (compares our score with our OOF expectation);
- paired SE between two of **our** submissions ~0.1 pp (component error correlation 0.93-0.98): a difference
  below ~0.2 pp between submission 1 and 2 is not interpretable; hex6 (-0.03) and seed members (-0.03..-0.05)
  are invisible on the board but still positive in expectation;
- between teams ~0.2 pp: Phase I gaps (0.08 to 1st) were noise.

## 2. Submission 1 = hybrid B, submitted as soon as Try-out passes (target 11/03-04)

Rationale: S is one scalar and all candidates are highly correlated, so the information value of submission 1
does not depend on which candidate we send; send the one with the best paired evidence (B: -0.090 vs the Phase I
six on 1,778 matched rows, 5/5 folds vs recipe A) and buy maximum time for the adaptive branch. Early
submission costs nothing (best of two counts); its value is failure detection with time to repair.

Expected S1 if Phase II test ~ Phase I test ~ training distribution: OOF_B 8.877, plus selection optimism
(+0.05..0.10, recipe A was chosen on these folds), plus the Phase I offset observed for the Phase-1 model
(test 8.70 vs OOF 9.00, SE 1.0) -> central 8.7-8.9, sampling +-0.3 -> **expected band 8.3-9.3**.

Before upload (reliability, since one crash among 1000 cases voids the evaluation):
- add a never-crash guard to the entrypoint: on any exception in the recipe-A path fall back to the Phase-1
  prediction; on any exception in the Phase-1 path fall back to recipe A; log the fallback; rebuild, re-run the
  pod n60 test and the build-box container test, re-hash;
- Try-out on one training image on the platform (T4): exec < 300 s, output equals the pod test value.

## 3. Reading S1 and the pre-committed action (decided now)

| Band | Condition | Interpretation | Submission 2 |
|---|---|---|---|
| 1 | S1 <= 9.3 and all cases succeeded | no material shift; internal paired estimates trusted | **B + hex6 (+ SSL-seed members if the LOFO gate passes)**; expected -0.03..-0.08 paired; package frozen 11/09, Try-out, submit 11/10-12 |
| 2a | 9.3 < S1 <= 10.5 and other teams' Phase II - Phase I shift is similar (>= +0.4) | test harder for everyone; ranking logic unchanged | same as band 1 |
| 2b | 9.3 < S1 <= 10.5 and others did not shift | our pipeline is shift-sensitive | **robust composition** chosen from the robustness audit (Section 4): the candidate that loses least under the perturbation family that hurts B most; drop fitted scalars if they hurt under shift |
| 3 | any case failed, timeout, or S1 > 10.5 (or < 7.5) | pipeline/format defect or extreme shift | **repair, not improve**: minimal fix of B, Try-out, resubmit; no new components |

Why band 1 is "incremental" and not "diverse": with paired SD ~0.1 pp and all alternatives expected worse than
B by >= 0.07, E[max(S_B, S_alt)] - S_B ~ +0.01, whereas B + hex6 (+seeds) is -0.03..-0.08 with ~90 %
probability of not being worse. Diversity only pays when candidates are close in expectation and weakly
correlated; ours are neither.

Why not keep submission 2 unused as insurance: failure of submission 1 is known within hours of evaluation;
insurance is only needed in band 3, which is observed before submission 2 is spent.

## 4. What to prepare now (GPU work with a decision attached)

1. **Robustness audit (band 2b table)**: on fold-4 OOF, perturb images with acquisition-like shifts (gamma
   +-, contrast, Gaussian blur, additive noise, +-5 % scale, mild JPEG) and record paired deltas for recipe A
   (with/without scalars), Phase-1 six, B at w=0.33/0.5/0.67, B+hex6. Output: for each perturbation, the
   composition that degrades least. This table is the only thing that makes band 2b actionable in time.
2. **Green confirmation -> hex6 deployment** per the fixed rule (nine+three / nine+one / nine frozen).
3. **SSL-seed members**: fold-0 SSL1004 regression + TTA, fold-4 SSL1005; LOFO gate >= 0.03 on 2 folds ->
   full-data retrain for band 1.
4. **Reliability tests**: odd inputs (non-972x1296, 3-channel, uint16, near-black) through the container.

## 5. Rationale audit of the work so far (what holds, what is weak)

Holds: fold-clean OOF with LOFO for every fitted parameter; pre-registered gates; paired bootstrap; the hybrid's
gain comes from a genuinely different view (whole-frame vs crops) and reproduces on 5/5 folds; the external
pre-only detector helped for the predicted reason (independence from the 20 shared annotation images).

Weak: (a) selection optimism (recipe A chosen on the same folds; Phase-1 OOF used validation-selected
checkpoints); (b) patient-level fold independence unverified; (c) no T4 measurement until Try-out; (d) effort
spent on gains (-0.03) below the paired noise floor (0.1) and far below the between-team floor (0.2).

Ceiling: component error correlation 0.93-0.98 means the remaining error is shared, not model-specific. A rough
label-noise floor supports this: if a technician's ROI holds ~100-150 cells, binomial sampling alone gives HEX
SD ~0.04-0.05 (7-8 % relative) and CV ~8 % relative; our OOF errors are CV 10.2 / HEX 10.1 / CD 6.3. Hence
(i) incremental modelling gains are structurally small, (ii) top-team ranking is noise-dominated, (iii) the
highest-value properties of submission 2 are reliability and robustness, not additional members.

## 6. Frameworks this plan follows

- Value of information (Raiffa & Schlaifer; Howard): information is worth collecting only if some branch
  changes the action -> Section 3 is written as bands with pre-committed actions.
- Decision Quality chain (Spetzler, Winter, Meyer 2016): frame, alternatives, information, values, reasoning,
  commitment -> the earlier plan had alternatives and information but no frame for what S1 could change.
- Pre-registration / forking paths (Gelman & Loken 2013); leaderboard reuse (Blum & Hardt 2015 "The Ladder";
  Dwork et al. 2015 reusable holdout) -> thresholds fixed before unblinding, no tuning on the board.
- Competition practice ("The Kaggle Book", Banachewicz & Massaron 2022: trust CV; make the two final picks
  serve different purposes); Klein's premortem; Duke "Thinking in Bets" (judge decisions, not outcomes).
