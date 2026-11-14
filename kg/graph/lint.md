# Lint report

Generated 2026-11-13T22:52:05

Nodes: constraint=36, dataset=8, decision=21, experiment=225, finding=66, idea=63, method=62, paper=75; stubs=0

Edges: 2221

## Errors (blocking)

none


## Warnings

none


## Info (tracked, not errors)

no-typed-edges (anchored to its source document)=12, ambiguous-number=41, experiment-without-delta_num=7

- **ambiguous-number** `wiki/experiments/exp-bb-geometric-augmentation.md` The lane summary gives 'fold 0, epochs 8-9, +0.116' as a member read without the baseline value or the metric (mean MAPE by the section's c…
- **ambiguous-number** `wiki/experiments/exp-bb-shading-correction.md` The unit and the metric of the offset (0.0825 -> 0.0959) are not stated in the notes.
- **ambiguous-number** `wiki/experiments/exp-bb-target-denoising-soft-teacher.md` Run date not stated in the lane summary (the read falls between the 11-09 and 11-13 ingests); the node's date is the summary's date. The pl…
- **ambiguous-number** `wiki/experiments/exp-bb-target-row-reweighting.md` Run date not stated in the lane summary (the read falls between the 11-09 and 11-13 ingests); the node's date is the summary's date. Baseli…
- **ambiguous-number** `wiki/experiments/exp-bb-vitl16-v2-regularised.md` v1 reference given as '~9.3'; delta is approximate.
- **ambiguous-number** `wiki/experiments/exp-cd-dinov3-distribution-head.md` Only '+0.008 five folds' is recorded (final_summary_cd section 4b, FINAL_SUMMARY section 3.2); the baseline value, the read date and whethe…
- **ambiguous-number** `wiki/experiments/exp-cd-family-screen-f04.md` 'vs B' value not printed; 6.185 + 0.079 = 6.264 (the five-fold value), while line 2 gives 6.269 / 6.263 on folds 0,4. Re-read 2026-11-09 (a…
- **ambiguous-number** `wiki/experiments/exp-cd-from-scratch-search.md` 'folds mixed' for the forward selection is as written in the ledger; whether 6.1733 is nested is not stated.
- **ambiguous-number** `wiki/experiments/exp-cd-stored-members.md` Same 'vs B' ambiguity as exp-cd-family-screen-f04. Re-read 2026-11-09 (arithmetic): 6.185 + 0.079 and 6.229 + 0.035 both land in 6.263-6.26…
- **ambiguous-number** `wiki/experiments/exp-cd-vith16-member.md` Run date not stated in the lane summary (the read falls between the 11-09 and 11-13 ingests); the node's date is the summary's date. The CD…
- **ambiguous-number** `wiki/experiments/exp-cd-vitl16-seed47.md` Sources disagree on what the second ViT-L seed is worth: RESULTS.tsv line 64 (11-09) reads 0.000 on folds 0,1,2,4; final_summary_cd section…
- **ambiguous-number** `wiki/experiments/exp-cv-area-sd-member.md` The lane summary gives '-0.0358 vs -0.0411 (worse)' without naming what the two deltas are measured against (read here as stack deltas agai…
- **ambiguous-number** `wiki/experiments/exp-cv-order-loss-other-members.md` The lane summary gives 'weight 0 / weight 0 / stack -0.0389 vs -0.0460' without naming the reference of the two stack deltas or the fold se…
- **ambiguous-number** `wiki/experiments/exp-gpu0-b2019-hyphen-map.md` Fold set not stated. Re-read 2026-11-09: the 'Reads so far' table still gives no folds; the A1 section's base is the best pipeline OOF (mea…
- **ambiguous-number** `wiki/experiments/exp-gpu0-minus22-correction.md` Fold set not stated in the report table (corrections on stored OOF). Re-read 2026-11-09: the 'Reads so far' table still gives no folds; the…
- **ambiguous-number** `wiki/experiments/exp-hex-cellroute-closed-variants.md` Baselines of the +0.007 and +0.006 HEX reads and the CV lane best value are not stated in the lane notes; RESULTS.tsv on the pod was not re…
- **ambiguous-number** `wiki/experiments/exp-hex-cellroute-step-on-tiers.md` The verifier line (O-VERIFY-HEX-TIERS) quotes 'kept version 9.70902 / 9.72080 (as computed)' for t2 / t3, while O-DEPLOY-014 quotes 'as com…
- **ambiguous-number** `wiki/experiments/exp-hex-from-scratch-forward-selection.md` The lane summary gives '9.74827 vs chain 9.73258 / ridge 9.72621' without the convention of 9.74827 or of the ridge value (the 38-column ri…
- **ambiguous-number** `wiki/experiments/exp-hex-per-source-stretch.md` Folds and baseline value not stated in the REPORT line.
- **ambiguous-number** `wiki/experiments/exp-hist-b-mix-seeds.md` Folds not stated; stored members only.
- **ambiguous-number** `wiki/experiments/exp-hist-canonical-zoom.md` Folds and baseline not stated; audit rates the evidence weak.
- **ambiguous-number** `wiki/experiments/exp-hist-cd-density-channel.md` Listed as closed in LANES; the number, folds and baseline are not in the sources read (archives deleted 11-06).
- **ambiguous-number** `wiki/experiments/exp-hist-cd-point-map-grid.md` Closed in LANES; numbers not in the sources read.
- **ambiguous-number** `wiki/experiments/exp-hist-cell-count-windows.md` Plan row gives two numbers without folds or metric; recorded as written.
- **ambiguous-number** `wiki/experiments/exp-hist-consensus-voronoi-cv.md` Closed in LANES; numbers not in the sources read.
- **ambiguous-number** `wiki/experiments/exp-hist-cpsam-whole-frame.md` Dropped on measured cost; no score.
- **ambiguous-number** `wiki/experiments/exp-hist-cv-only-graph-loss.md` Plan row gives '-0.002 against -0.017' without folds; delta_num is the difference (+0.015, arithmetic).
- **ambiguous-number** `wiki/experiments/exp-hist-disagreement-calibration.md` Folds and baseline not stated.
- **ambiguous-number** `wiki/experiments/exp-hist-graph-repair.md` Closed in LANES; no number in the sources read.
- **ambiguous-number** `wiki/experiments/exp-hist-hex6-ckpt-marker-r.md` Closed in LANES; no number in the sources read. Audit: the detector with the lowest val-5 F1 gave the strongest hex6 gain.
- **ambiguous-number** `wiki/experiments/exp-hist-label-clean-training.md` Folds not stated.
- **ambiguous-number** `wiki/experiments/exp-hist-localised-pooling.md` Folds not stated. Audit: replaced the whole frame rather than adding to it; not rerun with legible windows (medium evidence).
- **ambiguous-number** `wiki/experiments/exp-hist-per-metric-hybrid-weight.md` Folds not stated. Redone per lane under the 11-08 rule (CD, HEX).
- **ambiguous-number** `wiki/experiments/exp-hist-region-from-image.md` IoU differs by source (0.42 in BRIEF vs 0.57-0.61 / 0.605); probably different window definitions (most peaks >= 0.5 vs legible window). No…
- **ambiguous-number** `wiki/experiments/exp-hist-schedule10.md` Fold 0 confirmation has no record (audit).
- **ambiguous-number** `wiki/experiments/exp-hist-simple-cv-estimators.md` Closed in LANES; no number in the sources read.
- **ambiguous-number** `wiki/experiments/exp-hist-tail-pooling.md` Plan says the run was pending; LANES lists it closed; one run exited rc 0 with a NaN score. No number in the sources read.
- **ambiguous-number** `wiki/experiments/exp-hist-tessellation-zoom-response.md` Numbers not in the sources read.
- **ambiguous-number** `wiki/experiments/exp-hist-watershed-graphs.md` Closed in LANES; no number in the sources read.
- **ambiguous-number** `wiki/experiments/exp-rc-variance-budget.md` Shares are of OUR residual variance, so HEX sums to 124 % (the oracle total exceeds ours). The sampling term matches BRIEF's independent-dr…
- **ambiguous-number** `wiki/experiments/exp-ref-best-pipeline-f04.md` CD on these rows depends on the stretch variant (6.269 with 4-fold stretch, 6.263 cross-fit); some CD rows say 'vs B' without the value.
- **experiment-without-delta_num** `wiki/experiments/exp-cd-cd-weighted-loss.md` exp-cd-cd-weighted-loss
- **experiment-without-delta_num** `wiki/experiments/exp-cd-full-data-builds.md` exp-cd-full-data-builds
- **experiment-without-delta_num** `wiki/experiments/exp-cv-watershed-boundaries.md` exp-cv-watershed-boundaries
- **experiment-without-delta_num** `wiki/experiments/exp-deploy-gc-jobs-full-unloading.md` exp-deploy-gc-jobs-full-unloading
- **experiment-without-delta_num** `wiki/experiments/exp-deploy-t4-fp16-forced-parity-20.md` exp-deploy-t4-fp16-forced-parity-20
- **experiment-without-delta_num** `wiki/experiments/exp-hist-ssl-duration.md` exp-hist-ssl-duration
- **experiment-without-delta_num** `wiki/experiments/exp-rc-pseudo-label-pool.md` exp-rc-pseudo-label-pool
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-closed-means-closed-as-tested.md` con-closed-means-closed-as-tested
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-compare-phase2-only.md` con-compare-phase2-only
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-cpu-pool.md` con-cpu-pool
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-freeze-box.md` con-freeze-box
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-lane-resources.md` con-lane-resources
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-name-baseline-and-folds.md` con-name-baseline-and-folds
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-no-runtime-caveats.md` con-no-runtime-caveats
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-pod-memory-one-training.md` con-pod-memory-one-training
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-pod-safety.md` con-pod-safety
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-screen-folds-0-4.md` con-screen-folds-0-4
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-submission-deadline-20261115.md` con-submission-deadline-20261115
- **no-typed-edges (anchored to its source document)** `wiki/constraints/con-three-versions-before-drop.md` con-three-versions-before-drop

## Orphans (no typed edge in or out)

12 nodes. Body mentions do not count.

- `con-closed-means-closed-as-tested` (constraint)
- `con-compare-phase2-only` (constraint)
- `con-cpu-pool` (constraint)
- `con-freeze-box` (constraint)
- `con-lane-resources` (constraint)
- `con-name-baseline-and-folds` (constraint)
- `con-no-runtime-caveats` (constraint)
- `con-pod-memory-one-training` (constraint)
- `con-pod-safety` (constraint)
- `con-screen-folds-0-4` (constraint)
- `con-submission-deadline-20261115` (constraint)
- `con-three-versions-before-drop` (constraint)

