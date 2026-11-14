# Purpose

Read this before ingesting anything or answering a question from the KG.

## Project

CLEAR-EC Grand Challenge, Phase II. Input: one corneal endothelium specular image (1296 x 972 px, 0.7716 um/px).
Output: CD, CV and HEX. Score: mean of the three per-metric MAPEs (zero labels not scored). Test: 1000 hidden images,
two submissions, best counts; submission 2 unused. 8953 labelled frames, 25 frames with expert cell markers,
an external Topcon set of 385 images with cell points. No human annotation ([[con-no-human-annotation]]).
The objective is first place ([[con-first-place-objective]]).

Dates (Eastern): **freeze 2026-11-11 18:00** (each metric's model picked by five-fold OOF, [[dec-012-freeze-pick-by-five-fold]]),
full-data rebuild 11-12, T4 Try-out by the PI 11-13, submission 11-14 (platform deadline 11-15 06:00 Toronto).
State on 11-13 (close-out ingest): freeze held; packages for tiers p and t3 built and tested; upload, Try-out and submission
are the PI's actions and had not happened at the last source read. BRIEF gives the deadline as 2026-11-15 06:00 Toronto, not
reconciled with the earlier 11-14 ([[con-submission-deadline-20261115]]).

## Questions this KG answers

1. Which levers can close each metric's gap to first place, and how much of the gap can each close ([[con-bottleneck-first]])?
2. Has this idea, or something close, already been tried on the same inputs? Against which baseline, on which folds,
   with what number ([[con-closed-means-closed-as-tested]])?
3. What does the literature say: endothelium segmentation and device readouts, cell foundation models and their
   adaptation, MAPE-consistent point prediction, noisy and imbalanced regression?
4. Which beliefs are still untested ([[con-audit-beliefs-first]])?
5. What will hold on test: OOF trust per metric, the HEX test gap ([[find-hex-test-gap]], [[find-oof-test-trust]])?
6. What deploys: full-data retrain in 12 h on one GPU ([[con-12h-full-data-retrain]]), package track after the freeze?

## Close-out 2026-11-13 (shared ledger lines 78-213, `docs/FINAL_SUMMARY_20261113.md`, five lane final summaries; lane RESULTS.tsv not re-read)

All values are five-fold OOF MAPE, strict (everything fitted without the scored fold), incumbent recomputed in the same run.

| | mean | CD | CV | HEX | Node |
|---|---:|---:|---:|---:|---|
| Submission 1, test (1,000 images) | 8.9675 | 6.250 | 10.143 | 10.510 | [[exp-ref-submission1]] |
| Frozen candidate, full | 8.6679 | 6.1743 | 10.1160 | 9.71342 | [[exp-ref-freeze-candidate-integrated]] |
| **Tier p (built, tested, first to upload)** | **8.6682** | 6.1737 | 10.1115 | 9.71943 | [[exp-deploy-tier-table]] |
| Tier t1 | 8.6662 | 6.1737 | 10.1115 | 9.71342 | [[exp-deploy-tier-table]] |
| Tier t2 | 8.6728 | 6.1875 | 10.1115 | 9.71943 | [[exp-deploy-tier-table]] |
| Tier t3 (built, tested, fallback) | 8.6861 | 6.2138 | 10.1138 | 9.73076 | [[exp-deploy-tier-table]] |

Leaderboard from the PI on 11-13 (test): 1st 8.6920, 2nd 8.7035, a new 3rd 8.7560; ours is still submission 1. Submission 2 is unused.

- **Freeze unchanged** ([[dec-016-freeze-candidates-20261111]]): everything read against it after 18:00 EST 11-11 closed without a change.
  Synthetic pretraining is closed on all three routes ([[dec-017-synthetic-pretraining-followup]],
  [[find-synthetic-pretraining-gains-vanish-in-stacks]]); the CD arms ordered after the 11-11 leaderboard are closed
  ([[dec-018-cd-hypotheses-after-leaderboard]], [[find-cd-no-cheap-posthoc-lever]]); what is left of the CD error sits in source P1
  and in the label tails ([[find-cd-p1-error-within-donor-label-scatter]]).
- **Packaging** ([[dec-019-package-tiers-20261112]], [[dec-020-t4-assumed-tier-p-t3-fallback]]): tiers by build argument; tier p =
  CD t1 + CV t1 + HEX t2 + cell-route on t2. The full candidate's extra CD weights buy nothing over t1 under strict nesting
  ([[find-cd-full-extra-weights-nothing-over-t1]]). T4 assumed: container default fp16 + one detector copy, strict-score impact
  +0.0066 / +0.0067 on 200 labelled rows; precision is not what moves the outputs, the kernel change is
  ([[find-kernel-change-not-precision-moves-outputs]]).
- **Runtime** ([[con-platform-runtime-300s-t4]]): limit 300 s per case. Tier p ran 287-288 s on one Colab T4 and 310.7-314.2 s on a
  second; t3 173-174 s; t1 323-332 s; GPU peak 4.33 GB. The platform's own T4 time is unmeasured. Try-out rule: p above 270 s exec
  time, or a refused model upload, means switch image and model to t3.
- **Upload files** (build box, not pushed, not uploaded; [[con-upload-and-submission-are-pi-actions]]): p image dfc08a448e58 with a
  15.06 GB model archive, t3 image 8853d77c1f5a with an 8.81 GB one; 17 of 17 container jobs passed for each as uid 10001
  ([[exp-deploy-buildbox-container-test-p]], [[exp-deploy-buildbox-container-test-t3]]). Risk, unverified: a 10 GB per-upload cap
  ([[find-upload-size-risk-p-model-archive]]). Seven packaging bugs were found only outside the pod
  ([[find-packaging-bugs-outside-pod-tests]], [[find-silent-fallback-constants-hide-failure]]).
- **GPUs**: taken down on the PI's order at 21:41 EST 11-13 ([[dec-021-gpu-shutdown-closeout-20261113]]); the order was lifted the
  same evening: a fold-0 training-setting screen and an SSL screen on DINOv3-B/16 were running at the last ledger line read
  (line 213; [[exp-cd-training-setting-screen-dinov3]], [[exp-cd-ssl-pretrain-screen-dinov3]]). Three CPU reads made for the PI
  after the summaries closed changed nothing ([[exp-cd-vitl-averaged-tracks-in-stack]], [[exp-cd-recipe-a-e9-12-window-slot]];
  a fellow-eye transfer was closed by the PI as a trick, [[exp-cd-fellow-eye-residual-transfer]]).
- **Open or only partly tested**: recipe A hyperparameters never varied ([[idea-recipe-a-hyperparameters-never-varied]]), the
  attention-pool column's second seed ([[idea-attention-pool-seed-audit]]), the HEX deploy base convention
  ([[idea-hex-deploy-base-convention]]), a test HEX label-mix shift ([[idea-test-hex-label-mix-shift]]), a quartet-free CV stack
  ([[idea-cv-quartet-free-stack-shared-flips]]), the p model archive under 10 GB ([[idea-p-model-archive-under-10gb]]).
  Unverified: `cellroute_deploy_{t2,t3}.json` against a no-outer-fold refit (the p file's kept-version fit was recomputed on
  11-13 and reproduces to 2.8e-17; its version choice was not rerun).
- **Not ingested**: five analysis reports of 11-13 (hyperparameters, data, post-processing, bias, CD literature), written by
  separate agents after the consolidated summary. Row counts: the lane summaries write 8,954 raw / 8,953 labelled / 8,952 scored
  and trained; the consolidated summary as revised at 22:19 EST adds the waterfall (9,000 downloaded, 8,954 cached, 46 labelled
  frames lost to an exact-string ID match, 8,952 in the whole-frame full-data fits, 8,998 in the Phase-1 crop fits) and calls
  8,953 a prose error ([[ds-clear-ec-train-8953]], [[find-46-labelled-frames-lost-to-id-match]]).

The sections below are history: the freeze table of 11-11 (still the candidate) and the 11-09 stance.

## Freeze 2026-11-11 18:00 EST (from the shared ledger `ledger_20261111.jsonl`; lane RESULTS.tsv not re-read)

Candidate, five folds ([[dec-016-freeze-candidates-20261111]], [[exp-ref-freeze-candidate-integrated]]):

| Metric | Candidate | As computed | Strict | Best pipeline as computed / scalars refit | Node |
|---|---|---:|---:|---|---|
| CD | D-CD-001 | 6.17413 | 6.17428 | 6.26387 / 6.26103 | [[exp-cd-current-stack]] |
| CV | D-CV-001, 6 components | 10.11014 | 10.11596 | 10.15119 / 10.14786 | [[exp-cv-lane-best-stack]] |
| HEX | D-HEX-004, 39-column joint ridge | 9.71882 | 9.71977 | 9.80262 / 9.79992 | [[exp-hex-joint-nested-ridge]] |
| HEX + cell-route step | D-HEX-005 | 9.70581 | 9.71342 (end to end) | | [[exp-hex-cellroute-konan-step]] |
| Mean | verifier | 8.66336 | 8.66789 | 8.73923 / 8.73627 | [[exp-ref-freeze-candidate-integrated]] |

- The three lane stacks are now combined in one deploy entry ([[exp-ref-deploy-size-runtime]]): 55.42 GB fp32, about 30.0 GB with lossless savings.
- Framework: shared ledger, decision ids, lessons L1-L7 as gates, verifier ([[dec-015-framework-revision-20261111]]).
- The test gap is inside sampling noise ([[find-test-gap-within-sampling-noise]]); source shift is a level offset ([[find-source-shift-level-offset]]); a member better alone does not get stack weight ([[find-better-alone-not-stack-weight]]).
- Running outside the freeze: synthetic pretraining on ViT-L and recipe A ([[dec-017-synthetic-pretraining-followup]]).
- Open for the PI: container / model size and per-job time limits (ledger Q-001), docker build approval, use of the second submission.

The section below is the 11-09 stance, kept as history; its lane table is superseded by the table above.

## Current stance (snapshot 2026-11-09 15:09 UTC; every delta with its baseline and folds)

Leaderboard 2026-11-07: V14 8.6920 (CD 6.049, CV 9.928, HEX 10.099); ours, submission 1 (hybrid B), 8.9675
(6.250, 10.143, 10.510) = [[exp-ref-submission1]]. Test minus OOF for submission 1: CD -0.062, CV -0.061, HEX +0.395.

Best pipeline, five-fold OOF: mean 8.739; CD 6.264, CV 10.151, HEX 9.803 = [[exp-ref-best-pipeline-5fold]].
Screen rows (folds 0 and 4): mean 8.7578 = [[exp-ref-best-pipeline-f04]].

| Lane | Target (five-fold OOF) | Lane best now | Against | Node |
|---|---|---|---|---|
| CD | <= 6.11 (stretch 6.00) | 6.1775, five folds (ViT-L s47 adds 0.000 on folds 0,1,2,4; s48 dropped) | -0.086 vs best pipeline 6.264 | [[exp-cd-current-stack]] |
| CV | <= 9.99 (stretch 9.85) | 10.114, five folds, one recipe (DINOv3 seeds 46+47+49 on every fold; replaces the withdrawn 10.111 / -0.040) | -0.037 vs best pipeline 10.151 | [[exp-cv-lane-best-stack]] |
| HEX | <= 9.70 (stretch 9.55) | 9.7380, five folds (9.8704 on folds 0,4) | -0.065 vs best pipeline 9.8026 | [[exp-hex-main-chain]] |
| Backbone | one member, folds 0+4 <= 8.95 (recipe A 9.046) | ViT-L/16 kept as a member: alone 9.1146 (swa6), added -0.0131 vs best pipeline on folds 0,4; auxiliary detector targets neutral and stopped; ViT-L window added -0.0085 (folds 0,4), recipe A window running | target not met as a single model | [[exp-bb-vitl16-member]] |

The three lane stacks have not been combined into one pipeline; there is no measured combined OOF. Gaps to V14 per
metric if submission 1's test-minus-OOF repeats (LANES, 11-08): CV ~0.12, CD ~0.09, HEX ~0.04.

What the evidence says so far:
- Other encoder families add because they err differently ([[find-other-families-err-differently]]); on CD
  DINOv3-like members are saturated ([[find-cd-dinov3-like-members-saturated]]); ViT-L is the strongest family member.
- Mask area is a direct density read: calibrated mask CD is kept on CD ([[find-mask-cd-direct-density]]).
- CV and HEX residual variance already equals expert-centre error plus ~150-cell sampling (tentative, 25-frame oracle,
  [[find-residual-at-oracle-floor-cv-hex]]); the mask emulator and segmenter self-training are closed for CV and HEX
  ([[dec-014-close-mask-emulator-self-training]]).
- The device's outline rule is known: Konan weighted bisectors (k=2) read CV LOO-MAPE 7.64 with scale 1.09 on the 25
  marker frames (Voronoi 9.29, scale 2.08), but from detector peaks they add +0.015 to the CV stack: detection quality
  binds the cell route ([[find-konan-rule-explains-device-cv]], which qualifies the oracle-floor finding).
- Literature: CV/HEX grader noise exceeds the ~150-cell sampling floor ([[find-lit-cv-hex-grader-noise]]); a 66 % prediction
  spread at r 0.686 is expected ([[find-lit-cv-compression-expected]]); Topcon and Konan CV scales differ
  ([[find-lit-topcon-konan-cv-scale]]). Open ideas ranked by gap share: `wiki/_views/open-ideas-ranked.md`.
- Two-fold screens overstate gains ([[find-screen-folds-overstate]]); keep any OOF decrease, no CI gates
  ([[con-no-statistical-gates]]), never tune on reported rows ([[con-no-tuning-on-reported-rows]]).

Open PI decisions: [[dec-007-start-package-track]], [[dec-009-author-access-request]].

## Scope

In: every experiment and read from the lanes (11-08 on), the old lanes (11-06), the plan and audit, the root-cause
analysis, PI rules, and papers on endothelium imaging, cell segmentation and foundation models, counting/density,
MAPE and noisy-label regression. Out: patient-level data, Phase I scores as yardsticks.
