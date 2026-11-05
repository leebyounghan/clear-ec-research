# Green day 3 — ranked summary (2026-11-03 15:20 to 21:40 UTC)

**Directive:** continue green on all four GPUs; attack the per-cell-geometry bottleneck differently (parent). **Adoption
bar unchanged:** <= -0.02 pp mean on B + active hex6, fold-4 confirmation, T4-deployable, validated by 11-09. Every
experiment was pre-registered in `protocol.json` before its result; all CPU passes used the free-quota pool.

| rank | experiment | finding | effect | gate |
|---|---|---|---|---|
| 1 | **E4 error decomposition** (theory, PI-approved) | HEX loses to B at ~8 % random misses, ~0.17 x spacing jitter, ~14 % FPs; CV at ~2 % split-type FPs (robust to misses); CD has no headroom. Real detectors: miss 0.17-0.29, FP 0.16-0.28, jitter 0.20-0.25 (val-5); CPSAM miss 0.51. HEX is detection-accuracy-bound 2-3x off target; CV is FP-bound in simulation | decision map | n/a |
| 2 | **E5 split-aware robust CV** | tuned on the E4 grid only (merge 0.6, sliver 0.5, 10 % trimmed CV): beats B on the simulation up to 40 % FP; on the 8954 every source gives est. CV MAPE 12.7-14.1 with r_err 0.8 to B -> w = 0 | 0.0000 | fail; CV from points closed (real error is image-level, not split FPs) |
| 3 | **E7 hex6 threshold** | shipped 0.35 is the best of 0.20-0.50 (0.20 -0.0253, 0.50 -0.0194 vs -0.0311); the simulation's random-FP model does not transfer | +0.006 worse | fail |
| 4 | **E8 sharper detector targets** (sigma 5 / 3.5) | FP halves / quarters and jitter reaches 0.162 at sigma 3.5, but misses rise to 0.42; no point reaches misses <= 8 % and jitter <= 0.17 together; hex6 members +0.0001 | 0.0000 | fail |
| 5 | **E2 learned sides / area CNN** | srep validation fine (0.73 / r 0.93) but CLEAR-EC val-5 fails (0.65 / r 0.45); as members +both -0.0359 vs -0.0311 (gain -0.0048, 0.0002 short of evidence) | -0.0048 | fail (sanity + gate) |
| 6 | **E3 box-restricted std pooling** (research) | worse than the standard member on both folds (+0.071, +0.096 TTA), CD worst | +0.07..+0.10 | fail |
| 7 | **E1 CPSAM cells** | centroid F1 0.52 vs expert points; member +0.0006; mask-based CV / hexagonality w = 0 | 0.0000 | fail |
| 8 | **E6 predicted-area CV at CPSAM / consensus points** | w = 0 (est 13.3-13.6, r_err 0.8) | 0.0000 | fail |
| 9 | **E11b SREP + synthetic v5 mixed detector** (label-free, 2026-11-04) | on the 25 expert images miss 0.101 / FP 0.264 / jitter 0.253 / F1 0.865 vs shipped 0.150 / 0.165 / 0.215 / 0.862: recall passes the 0.12 rule but the synthetic FP / jitter profile carries over (FP +0.10 on 18/25, jitter +0.04 on 23/25) | n/a (detector gate) | fail |
| 10 | **E11a noisy student of the shipped detector** (label-free, 2026-11-04) | miss 0.196 / FP 0.095 / jitter 0.214 / F1 0.859: precision up on every image, misses +0.046 (22/25 worse), F1 unchanged | n/a (detector gate) | fail |
| 11 | **E10 synthetic-only detector** (route B, 2026-11-04) | miss 0.182 / FP 0.542 / jitter 0.309 / F1 0.744 on the 25 real images; renderer v1-v5 never matched the real FP / jitter profile | n/a (detector gate) | fail; route B closed |
| 12 | **E9 label verifier** (route A, CPU) | expert-box windows rank at the 30-38th percentile of random windows; beat-all-nulls 8.2 % vs chance 4.8 % | n/a | fail; route A closed |

## What day 3 established

1. The bottleneck is quantified: HEX needs recall >= 92 % with < 4 px localisation inside the measured region; every
   point source is at 71-83 % recall and 5-6 px. Sharpening the detector target moves along a recall-precision curve
   that never reaches both ends (E8); lowering the threshold does not help on real images (E7); segmentation (CPSAM)
   and appearance-based per-cell heads (E2) are further away.
2. CV cannot be recovered from points on this data: the robust estimator removes the simulated split-FP damage but the
   real CV error is image-level and 0.8-correlated with the regressor's (E5), and learned areas do not transfer (E6).
3. Localising the regressor's pooling to the measured window hurts (E3), completing the series roi_crop / local_pool /
   masked-pool.
4. The only green component with value remains the shipped hex6 blend (-0.0311), already at its best threshold and
   composition; the one documented refinement is the coverage-conditional weight (-0.0374, evidence-only).

## Label-free routes (2026-11-03 23:00 to 2026-11-04 14:00 UTC)

The PI ruled out human annotation, so the cell-level bridge had to come without labels. Route A (a verifier that could
pick annotation-quality windows, E9) is flat; route B (a detector trained on synthetic endothelium, E10) fails on real
images with FP 0.54; and the two label-free pushes of the shipped pre-only detector fail in opposite directions: the
noisy student (E11a) trades recall for precision, the SREP + synthetic mixed model (E11b) trades precision and jitter for
recall. Neither reaches the HEX requirement of recall >= 92 % at < 4 px. Verdicts: `e9_label_verifier/`, `e10_synth/`,
`e11_nolabel/verdict.md`. The first E11 attempt killed pod clear-ec-research-4gpu-1026 (cgroup OOM inside
`load_pseudo()`: per-row npz re-reads kept ~139 GB of copies); the rerun on clear-ec-research-4gpu-1104 with the
read-once fix used 6.6 GB for both trainings together.

## GPU use

15:34-21:33 UTC all four GPUs: E1 segmentation (2 GPUs, 55 min), E2 train/infer, E3 fold 0 (GPUs 0-1, 4.4 h) and fold 1
(GPUs 2-3, 4.4 h), E8 sigma 5 (lane01) and 3.5 (lane23), E3 TTA, E6 inference. Both lanes idle on keepalive from 21:35 UTC;
nothing further queued (no remaining green idea with a plausible >= 0.02 pp return; the next lever is in-domain labels).
2026-11-04: E10 training + score25 (lane01, 05:1x UTC), E11a pseudo (GPUs 0-1, 10 min) + student (GPU 0, 25 min), E11b mixed
(GPU 2, 31 min); lanes idle on keepalive from 13:48 UTC.

## Files

`protocol.json`; verdicts: `e1_cpsam/`, `e2_cellhead/`, `e3_boxpool/`, `e4_degradation/`, `e5_robust_cv/`, `e6_pred_area_cv/`, `e7_threshold/`, `e8_sigma/`, `e9_label_verifier/`, `e10_synth/`, `e11_nolabel/`; export for the parent: `experiments/submission2_search_20261103/green_features.npz`; ledger `experiments/green_track_20261028/ledger.json`.
