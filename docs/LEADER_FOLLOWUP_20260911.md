# Leader report and same-direction follow-up — September 11 UTC

## Evidence and protected candidate

User reported first place during the September 11 01:10 UTC session and asked
to keep developing this direction. Exact leaderboard score, evaluation URL,
and submitted image/model pairing have not yet been supplied or independently
verified. Do not invent those values or attribute the gain to crop selection
alone: the packaged candidate combines six members and scalar calibration.

Preserve `submission_candidate_20260909/` and its original two archives as the
rollback candidate. Do not overwrite its config, weights, inference source or
archive. The user-reported first-place submission still needs an exact artifact
association. No further platform upload or scored submission is performed here.

## Active work, not a new launch

At 01:11 UTC, `clear-ec-crop-screen-2gpu` was healthy, zero restarts, exactly two
H200 on one node. Both controls completed 5215 training updates and were in
terminal evaluation: fold0 1400/1791 images, fold1 1600/1791. `failed={}`.
Guarded fits follow these controls in the existing finite queue. No source,
resource limit, gate or controller has been changed.

See `CROP_DINO_SCREEN_20260910.md` for the frozen experiment and cleanup scope.
That heartbeat remains monitor/audit/cleanup only; this document does not extend
its authority to launch confirmation or other training.

## Ordered next decisions

1. Finish the matched original-versus-artifact-guard crop screen. Keep the
   fixed equal-weight add7 gate; no post-result weight search or threshold tuning.
2. If the gate passes, prepare a separately bounded two-GPU confirmation on
   remaining slide-group folds2–4 with both guarded and matched control fits.
   Lock recipe and acceptance criteria before seeing those results. Require
   improvement on the remaining-fold aggregate versus both current6 and control,
   as well as no worsening in any confirmation fold. All folds have historical
   reuse: this is not a new untouched test or proof of leaderboard improvement.
3. If confirmed, train one fresh all-data guarded member and validate native
   inference, precision and full-ensemble behavior in a NEW candidate directory.
   Compare uncalibrated decoded means consistently before any calibration decision.
4. If the screen fails, retain the current candidate. Do not automatically run
   more seeds or tune crop thresholds on these same scores. Review failure by
   pre-existing quality strata using aggregates only, then formulate one bounded
   new hypothesis. Do not assume that more restrictive crops are always better.
5. DINO/JEPA and additional external pretraining are not the immediate priority;
   DINO eligibility remains unresolved. Failed specialist recipes remain excluded.

## Provisional schedule, not a running confirmation queue

Measured current training is about2.99 seconds/update. One5215-update fit is
about4.33 GPU-hours, plus roughly12–13 minutes for1791-image40-view evaluation
and checkpoint I/O. Six remaining-fold control/guarded fits would therefore
require roughly27–30 GPU-hours, or14–16 elapsed hours on two GPUs with margin.
An all-data fit and native/runtime checks require additional time; allow another
6–10 hours before packaging, subject to measured throughput and actual recipe.

- September11: finish crop screen and decide whether confirmation is warranted.
- September11–12: conditional two-GPU confirmation; no simultaneous expansion.
- September12–13: conditional all-data member, integration and runtime checks.
- September13: target candidate freeze and package readiness, retaining rollback.
- September14: user-controlled final submission with upload/validation buffer.

These are calendar targets, not a verified competition deadline/timezone. If
confirmation or runtime checks miss the buffer, keep the existing successful
candidate instead of promoting an incompletely validated change. User report
does not establish the exact remaining submission count; confirm it before any
future scored submission. No additional scored submission is authorized here.
