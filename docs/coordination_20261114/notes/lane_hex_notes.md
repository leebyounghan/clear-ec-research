# HEX lane (GPU 1), notes for sibling sessions

Started 2026-11-07 20:45 EST. Pod folder `experiments/lane_hex_20261108/` (`RESULTS.tsv`, `REPORT.md`, `source/`, `feats/`, `preds/`, `logs/`).
Reference: best pipeline HEX 9.8026 (5 folds), 9.9079 (folds 0+4).

## Status, 2026-11-07 23:10 EST

HEX chain (`source/chain.py`, nested choice per outer fold on the other four folds, stretch on the other four folds):
9.7693 on five folds (-0.033), 9.8856 on folds 0+4 (-0.022). Kept steps: A/Phase-1 weight 0.7, hex6 weight 0.28 -> 0.16-0.22,
10-seed pre47/pre48 graph ensemble, ridge on 4 whole-image statistics, count-matched hex6 from the CD spacing prior.
Candidate and final HEX: `preds/chain_try5.npz`.

Possibly useful to the CD and CV lanes: a ridge on 4 whole-image statistics (mean, contrast, black fraction, Laplacian focus,
from the raw frame at half size; `experiments/eval_harness_20261031/image_stats.npz`, row order = base idx) on the log residual
gave HEX -0.010 (folds 0+4) and -0.013 (five folds) on top of everything else. Deployable: computed from the image alone.
25 richer statistics (`feats/img_rich.npz`: vignetting, quadrant means, local contrast, focus by region) did the same, not more.
HEX bias by imaging year runs from -1.3 % (2021) to +0.3 % (2023); image statistics stand in for year at test time.

Dropped: CD/CV predictions as HEX features, MAPE-loss linear fit on uncertainty features, detector legibility counts, consensus
point features, quadratic term, global graph scale.

## GPU 1

Coordinator OK (2026-11-07 23:00 EST): up to 2 small graph jobs next to CPSAM fold 4. Runner `source/queue_gpu1_v2.sh`, jobs in
`jobs_gpu1.tsv`, concurrency in `slots_gpu1.txt`. d64 5-layer graph network: 75 s per seed-fold, ~31 min per 5x5 job.
`source/gnn_hex.py --ctxfile FILE` adds row-ordered features (npz with `idx` and `F` or named columns) to the graph network head.

2026-11-08 00:05 EST: the 4 image statistics in the graph network head (`--ctxfile .../image_stats.npz`, d64 l5, 5 seeds,
pre46 graphs) gave, half-and-half with the best pipeline's graph part: HEX -0.016 (folds 0+4), -0.023 (five folds); the
same network without them: -0.0002 / +0.004. On top of the HEX chain it adds nothing: it is the same signal as the image
statistics ridge. CV does not gain from it: CV half-and-half +0.007 / +0.003, the same as the control (+0.005 / +0.004).
Ridge evidence (`source/feat_ridge.py`, `feats/ridge_k3.npz`, `feats/k3_ridge_graph.npz` = pre46 k3 graph + 5 node columns)
added -0.002 / -0.001 to the HEX chain. Chain now 9.7680 (five folds), 9.8834 (folds 0+4): `preds/chain_main.npz`.
CPSAM fold 4 epoch 5 took 63 min with two of my jobs beside it, so GPU 1 runs one HEX job until CPSAM ends.

2026-11-08 04:10 EST: zero-shot Cellpose-SAM segmentation (local `cpsam_models/cpsam_v2`, no training; `source/seg_hex.py`,
`source/seg_run.py`). On the same 100 random frames, loose thresholds (flow 0.8, cellprob -2) + CLAHE vs default:
usable frames 85 vs 53; hex6 from masks r with HEX +0.665 (Voronoi hex6 on the same rows +0.601); r with the HEX chain
residual +0.250 (Voronoi +0.159). Possibly useful to the CV lane: mask area CV r with CV +0.45, mask hex6 r with CV -0.54.
Full run on all 8954 frames started on GPU 1 at duty 0.35 (`seg_loose/cNNN.npz`, masks kept as uint16 so features can be
recomputed on CPU). It replaced the multi-detector graph job (log in `logs/archive/`); one graph job stays beside it.

2026-11-08 05:35 EST: the HEX lane looks close to its ceiling. HEX chain 9.7683 (five folds, -0.034), 9.8808 (folds 0+4,
-0.027) with a HEX-only graph network mixed in (`preds/chain_gnn2.npz`). On 1024 frames, the full-frame segmentation's
hex6 tracks HEX like Voronoi hex6 (r 0.47) and not the chain residual (r -0.06). The 85-frame +0.25 was noise.
On the 25 marker frames, marker hex6 r 0.84 vs hybrid B 0.81; mask hex6 inside the technician's box 0.77 vs whole frame
0.73. Stretch per label source (ID format, or image classifier AUC 0.999 from 25 image statistics; probabilities in
`feats/src_image_prob.npy`) made HEX worse: both sources have the same residual mean and SD.
GPU 1: segmentation run (duty 0.5) + one graph job (duty 0.25); CPSAM ~57 min per epoch, epoch 11 of 20, so it ends
around 14:00 EST, not 10:00. `source/post_cpsam.sh` lifts the throttles when its pid exits.

## 2026-11-08 06:55 EST (HEX lane)
- HEX chain: 9.7640 on five folds (-0.039 against the best pipeline), 9.8783 on folds 0+4 (-0.030). New kept step: ridge evidence from the 16-detector consensus graphs. Predictions `preds/chain_gnn4.npz`.
- Closed: multi-detector graph network; ridge evidence from other detector graphs (pre47, pre48, frozen pre46, mean of 12). On top of the new chain, imgrich, family, gscale and fuse16 all raise f04.
- GPU 1 is now HEX-only (CPSAM stopped 11:40Z). Graph jobs take about 32 min each. Segmentation of all frames ends about 09:45 EST; mask-graph jobs follow automatically.
- Ready for the backbone lane's CPSAM 512: chain step `cpsam` mixes its HEX (epochs 8-12 / 10-12 / 12, share 0.1-0.5) into the HEX base on folds 0 and 4. I'll read it when fold 0 lands (about 00:00 EST 11-09).

## 2026-11-08 08:00 EST (HEX lane)
- GPU 1 runs HEX-weighted CPSAM 512 (loss 0.5/0.5/2 on CD/CV/HEX, `source/hexcps.py` over a frozen recipe copy), SSL fold 4 first (backbone fold-4 SSL checkpoint, checked fold-4-train-only). ETA about 16:00 EST.
- Coordinator decision (a): `source/cps_queue.sh` then runs no-SSL folds 4, 0, 1, 2, 3 (about 8 h each). Folds 0+4 read after fold 0 (about 08:00 EST 11-09). If HEX does not go down, `STOP_CPS_QUEUE` stops the queue before fold 1. Five folds about 08:00 EST 11-10. Fold 1-3 split files checked equal to the chain folds.
- GPU sharing: CPSAM runs 5.32 s/update alone and 12.96 with two graph jobs beside it. Graph queue paused (two jobs SIGSTOPped, resume after CPSAM). Segmentation throttled to duty 0.25: 5.79 s/update (+9%).
- Backbone CPSAM fold 0 is at epoch 4 (08:00 EST); at about 1 h per epoch, epoch 12 lands about 16:00 EST, not 10:30. The chain reads it then: `cpsam` (share chosen on the other fold, stretch on folds 1-3) and `cpsam_fix` (share 0.2).
- HEX chain unchanged: 9.7640 five folds, 9.8783 folds 0+4. 10-seed HEX graph net is worse than 5 seeds (f04 9.8809), so not used.

## 11:50 EST 11-08 (HEX lane)
- Chain kept a segmentation step: HEX 9.7405 five folds (-0.062 vs best pipeline), 9.8709 folds 0+4 (-0.037). preds/chain_segm.npz.
  Step `segm`: watershed segmentation (loose + CLAHE) of all 8954 frames, ridge (alpha 100) on 16 cell-shape features, fit and applied only on frames with >= 50 interior cells (7704). Fold 0 alone +0.021, folds 1-4 -0.02 to -0.06.
- Dropped seg versions: median-filled (+0.026), h6 blend by cell count (+0.0002 on top), boosted trees (+0.009).
- For CD/CV lanes, FYI: experiments/lane_hex_20261108/preds/seg_all.npz has per-frame cd_mm2 and area_cv from the same segmentation. Raw r with the best pipeline's log residual is small (CD -0.03, CV +0.03 on 7704 frames), so probably little there.
- GPU 1: a graph job beside CPSAM slowed it to 9.12 s/update (+71 %), so it was stopped; CPSAM runs alone (5.33). The mask-graph network now runs on CPU (3 threads).
- Housekeeping: the failed first launch of HEX-weighted CPSAM fold 4 (wrong python) was moved to archive/hexcps_c512_f4_failed_launch_wrong_python_20261108 (+ log archive/hexcps_f4_failed_launch.log); nothing deleted.

## 17:10 EST 11-08 (HEX lane): CPSAM loss switched to multi-target
- HEX-weighted CPSAM 512 (SSL) fold 4 trails the backbone CPSAM fold 4. Share 0.2 on the current main: fold-4 HEX -0.009 vs -0.021. Standalone HEX epoch 12: 10.44 vs 10.19 (epochs 8-12 mean 10.34 vs 10.24).
- Per the coordinator's instruction, the no-SSL runs now use the recipe's multi-target loss (source/mtcps.py), queue source/cps_queue2.sh, folds 4, 0, 1, 2, 3. Fold 4 started 17:04 EST, about 8.7 h per fold.
- The HEX-weighted no-SSL fold 4 was stopped after 10 min (pid 214214, cmdline checked; old queue ended via STOP_CPS_QUEUE). Moved, not deleted:
  - archive/hexcps_nossl_c512_f4_stopped_switch_to_multitarget_20261108
  - archive/hexcps_nossl_f4_stopped.log
- Main chain: HEX 9.7408 five folds / 9.8706 folds 0+4 (+ segv, a Delaunay neighbour-count step from the seg masks, -0.0003).
- A second segmentation run (frames rotated 90 deg) runs at duty 0.25 beside CPSAM (+9.6 %), so the two runs can be averaged.
- Per-image runtime is no longer measured or reported, as instructed.

## 17:40 EST 11-08 (HEX lane): following the new charter section
- Main chain is now the short one: h6w, imgstats, prior, segm. HEX 9.7428 five folds / 9.8748 folds 0+4 (preds/chain_simple4.npz).
  The 9-step chain (9.7405 / 9.8709) is kept only as a reference: its five extra steps together give -0.002 / -0.004.
- Stopped:
  - the graph queue on GPU 1 (STOP_QUEUE, slots 0);
  - the two paused graph-net jobs (pids 177058, 179619, cmdline checked);
  - the tiny chain steps (segv dropped).
- CPSAM reads now use the fixed share 0.2 only (mtnossl_fix after fold 4 and 0, mtcnn5_fix for five folds). No share grids.
- Directions table with continue/stop is in REPORT.md. I will update it at 09:00 EST daily.

## 20:35 EST 11-08 (HEX lane): main chain restored
- Per the coordinator's correction, the main is the full kept chain again: HEX 9.7405 five folds / 9.8709 folds 0+4 (preds/chain_segm.npz).
- All waiters read on it: segm2/segr (in place of segm, after the rotated segmentation run), CPSAM mtnossl_fix and cpsam_fix at share 0.2.
- segv (Delaunay counts, -0.0003 on folds 0+4 / +0.0003 on five folds) is left out, matching the numbers given.

### 12:45 EST 11-11 — HEX lane → coordinator (my outbound messages are paused by the app's 10-message cap until the user types in my session)
- mtcnn5 s47 fold 1 stopped by verified pid (GPU 2 now free, 0 MiB); the backbone lane was told. Folds 0 and 4 continue: fold 4 about 14:05 EST, fold 0 about 15:10 EST (slow card); the two-fold seed read follows (if fold 0 is not done at 15:00, I report with the epochs that exist).
- Freeze candidate unchanged: 39-column joint ridge, strict 9.71977 (`preds/chain_joint_attnp.npz`, `preds/deploy_params_joint_attnp.json`). Synthetic zero-shot columns: strict 9.72086 (+0.00109), closed.
- Synthetic-pretrained folds 0+4 member: when its predictions are in the notes I will read it inside the 39-column ridge, strict, against 9.71977. A member on folds 0 and 4 only cannot be a full column (each outer fold's training rows lack it on three folds), so it enters as the member's log deviation from the blend on its folds and 0 elsewhere, plus an availability flag; deltas reported on folds 0+4 and five folds.
- Cell-route refit on `chain_joint_attnp.npz`: not posted yet in its notes.
