# HEX lane (GPU 1), report

Target: 5-fold OOF HEX <= 9.70, stretch 9.55. Reference: best pipeline HEX 9.8026 (five folds), 9.9079 (folds 0 and 4).
One line per try in `RESULTS.tsv`.

## Status, 2026-11-11 09:45 EST

HEX freeze chain after the seed audit (10 steps: ap1 h6w graphens imgstats prior ridgeev ridgeev_fall16 segm cvw3m_fix mtcnn5_fix):
**9.73258** on five folds (-0.0700 against the best pipeline), **9.86437** on folds 0 and 4 (-0.0435). With the cell-route step on top:
9.72623 / 9.8554. Nested estimate of the chain procedure 9.74255 (strict 9.74961): the honest HEX level is about 9.74. Coordinator 10:50 EST: the joint nested ridge is the HEX freeze candidate; since 11:40 EST the 39-column version with the attention-pool column (strict 9.71977; 38 columns 9.72662 as fallback 1); its deploy fit is the nested procedure with no outer fold (FREEZE_CANDIDATE.md); the chain is the fallback. CD and CV unchanged. Predictions: `preds/chain_seedaudit10.npz`; deploy params `preds/deploy_params.json`
(parity 8.9e-16); all full-data builds done, including cvw3 s47. Freeze candidate: `FREEZE_CANDIDATE.md`.

## Deploy gaps from the integration rehearsal (11-10; status as of 11-11 01:10 EST: all builds done)

| # | Gap | Status |
|---|---|---|
| 1 | Graph members G_s10x3, G_hex46: all-rows mode and weights | done: `deploy/graph_models/full_d64_l5_pre47_s10`, `full_d64_l5_pre48_s10` (10 seeds), `full_hex_pre46_s5` (5 seeds); 803 / 811 / 401 s on GPU 1; `gnn_hex_infer --check` max diff 1.2-1.3e-8 against each in-sample prediction. noctx_s10 = the CV lane's gfull_noctx |
| 2 | Feature builders bound to training frames; prior read OOF CD | done: `source/hex_test_features.py` (images + peaks + hybrid-B CD = 0.5 A + 0.5 P1 -> image statistics, prior, ridge k3, ridge fall16, segmentation, 4 cell graphs) and `source/det_peaks_image.py` (full-data detectors -> peaks). Parity on 40 training frames used as pseudo-test frames: all outputs equal the stored training features (prior to 1e-6, float32 storage); detector peaks equal the stored fold-0 peaks for all three detector families (12 frames each); segmentation features within 1.1e-4 (646 values, Cellpose on GPU). In-memory entry for the integrated inference: `hex_test_features.build_arrays(imgs, peaks, cd, outd)` (single-view peaks) |
| 3 | cvw3_fix full-data model | done: `deploy/cvw3_f99_s46/runs/fold99/dinov3_b16/epoch_{08..12}.pt` (2500 s on GPU 1); inference `family_infer.py` (exact on fold checkpoints); 64-frame sanity: r 0.978 with the OOF member |
| 4 | Detectors for ridgeev_fall16 | 8, not 4 (the frozen heads had only fold versions). 4 sg2 done (385-396 s each), 4 frozen heads done (sg2f 195-205 s, sg2pref 670 s) |
| 5 | mtcnn5_fix five-fold read | kept (-0.0053 / -0.0022). Deploy params regenerated (nested parity 9.7358 exact, 8.9e-16); cell-route refit done (9.7283); full-data CPSAM done (`deploy/mtcps_c512_f99/runs/fold99/cpsam_c512/epoch_{08..12}.pt`, 36079 s = 10.0 h, GPU alone; 64-frame sanity HEX r 0.954 with the OOF member, mean log ratio -0.0072); its inference `family_infer.py` verified exact on the fold-4 checkpoints |
| 6 | CV lane's all-rows BEST and GRAPH | posted by the CV lane; mapping in FREEZE_CANDIDATE.md (graph = its GRAPH HEX column, best = graph + its sim ridge HEX) |
| 7 | Row count in FREEZE_CANDIDATE.md | fixed (8954); chain outputs now carry idx and fold |

## Directions, continue or stop (11-11 09:45 EST, written late for the 09:00 slot; freeze 11-11 18:00 EST)

| Direction | Box used | Result so far | Decision |
|---|---|---|---|
| Seed audit of the freeze chain (coordinator rule) | cvw3 s47 five folds + f99: about 4 GPU-h; gnn pre47: 0.5 GPU-h | cvw3 stays as the seed mean; gnn:hex drops; chain 9.73258 (was 9.7358); cell-route refit 9.72623 | done; chain file, deploy params and FREEZE_CANDIDATE.md updated |
| mtcnn5 second seed (s47), two-fold check, then a four-fold line | folds 0, 4, 1, 2 on GPUs 3, 0, 2, 1 (about 8.6-9.4 h each) | running; fold 4 lands about 14:05 EST, fold 0 about 14:30-15:15 EST (its epochs vary 5.3-6.6 s/update; not caused by the newer folds: no CPU throttling, fold 4 flat at 5.30), folds 1 and 2 about 15:00 / 15:45 EST | continue; the two-fold read (folds 0+4) when fold 0 lands; it does not change the deploy model |
| Vigueras-Guillen TVST edge network (cells -> segm recipe) | about 0.6 GPU-h | HEX +0.0003 five folds / +0.0052 folds 0+4; CV lane +0.0196 | stopped (closed for HEX and CV) |
| Cross-lane members at share 0.2 | CPU minutes per read | backbone C-Mixup DINOv3-B +0.0153, attention-pool DINOv3-B +0.0055 (folds 0+4) | continue only if another lane finishes a member before the freeze |
| Deploy builds | | all done (detectors, graph nets, cvw3 s46 + s47, CPSAM) | done |
| Segmentation, HEX-weighted loss, graph-network variants | | | stopped (earlier rows) |

## The chain

How it is read (`source/chain.py`, PI rule of 2026-11-08): each step offers a few options; for outer fold f the option is
chosen by HEX on the other four folds, and the stretch is fit on the other four folds. A step is kept when it lowers HEX on
folds 0 and 4 against the previous step. Predictions: `preds/chain_main.npz` (candidate per fold and final HEX).

| Step, kept | Folds 0+4 | Five folds | Nested picks |
|---|---:|---:|---|
| Best pipeline | 9.9079 | 9.8026 | |
| A / Phase-1 weight in the HEX base | 9.9064 | 9.7935 | 0.7 (fold 4: 0.8) |
| hex6 weight 0.28 lowered | 9.9024 | 9.7855 | 0.16 (folds 3, 4: 0.22) |
| Graph HEX ensemble with 10-seed pre47/pre48 graphs | 9.9019 | 9.7856 | three 10-seed graphs |
| Ridge on 4 whole-image statistics (mean, contrast, black fraction, focus) | 9.8923 | 9.7729 | alpha 10 (folds 2, 3: 1000) |
| Count-matched hex6 from the CD spacing prior | 9.8856 | 9.7693 | cm1.0 clean hex6 |
| Ridge evidence: hex6 from side counts without short ridges, dark-ridge counts, ridge darkness | 9.8834 | 9.7680 | 5 features, alpha 1000 (fold 1: 10) |
| HEX-only graph network (pre46 graphs, loss on HEX alone, 5 seeds) mixed 0.25 into the graph part | 9.8808 | 9.7683 | mix 0.25 (fold 2: none) |
| Ridge evidence from the 16-detector consensus graphs (hex6 with ridges > 0.15 x median) | 9.8783 | 9.7640 | h6_s15 on all folds |
| Zero-shot Cellpose-SAM segmentation (local `cpsam_models/cpsam_v2`, no training; loose thresholds + CLAHE, all 8954 frames): ridge on 16 cell-shape features, fit and applied only on frames with >= 50 interior cells (7704) | 9.8709 | 9.7405 | alpha 100 on all folds (fold 0 alone +0.021) |
| CV lane DINOv3 b16 (CV-weighted loss, five folds) mixed into the HEX base at share 0.2 | 9.8704 | 9.7380 | fixed, no choice |

Segmentation, dropped: the same ridge with missing frames filled by the median (+0.026 on folds 0+4); segmentation hex6 as a
direct measurement blended by cell count (-0.004 on its own, +0.0002 on top of the kept step); boosted trees on the residual
(+0.009). The per-cell graph network on segmentation masks is running on CPU (it slows CPSAM by 71 % on GPU 1). A second segmentation
run with each frame rotated 90 degrees (`source/seg_run2.py`, same settings; frame-level h6 r 0.88 with the first run) runs
beside CPSAM at duty 0.25 (+9.6 %), about 10 h; the two runs will be averaged (`segm2`).

## HEX-weighted CPSAM 512 (approved by the coordinator 2026-11-08 07:10 EST)

Why: mixing the backbone lane's CPSAM 512 fold-4 HEX into this chain at share 0.2 lowers fold-4 HEX by about 0.020 (a look
on fold 4 only; the read waits for fold 0). No other open HEX idea moves it this much.
- Recipe: the backbone lane's family recipe, frozen copy `source/family_recipe_copy.py`, run through `source/hexcps.py`, which
  changes only the regression loss weights (CD/CV/HEX 0.5/0.5/2; equal weights reproduce the original loss exactly).
  12 epochs, schedule 10, seed 46, the same split files (folds 1-3 checked equal to the chain folds).
- SSL fold 4 done (22:00 UTC). It trails the backbone CPSAM on fold 4: standalone HEX at epoch 12 10.44 vs 10.19 (epochs
  8-12 mean 10.34 vs 10.24), and mixed into the chain at share 0.2 fold-4 HEX -0.009 vs -0.021. The HEX-weighted loss is dropped.
- Coordinator decision (a), then the loss switch: train without SSL on all five folds with the recipe's own multi-target loss
  (`source/mtcps.py`), queue `source/cps_queue2.sh`: folds 4, 0, 1, 2, 3, about 8.7 h each (fold 0 ends about 10:30 EST
  11-09, five folds about 12:30 EST 11-10). The HEX-weighted no-SSL fold 4 was stopped after 10 minutes and moved to
  `archive/`. After fold 4: no-SSL vs the backbone's SSL fold 4, same loss (the coordinator hears first if SSL wins clearly).
  After fold 0: `mtnossl` (share chosen on the other fold) and `mtnossl_fix` (0.2) on folds 0+4; if HEX does not go down,
  `STOP_CPS_QUEUE2` ends the queue. Five folds are read with `mtcnn5`.
- GPU sharing: with two small jobs beside it CPSAM ran at 12.96 s/update against 5.32 alone, so the graph queue is paused
  while CPSAM runs; segmentation is throttled.

All features are computed from the image and the detector peaks alone, so the chain deploys.
- Image statistics: raw frame at half size (`eval_harness_20261031/image_stats.npz` recipe). 25 richer statistics
  (`feats/img_rich.npz`) do the same job, not more.
- Ridge evidence (`source/feat_ridge.py`): Voronoi of the graph nodes in the three legible windows; per ridge its length
  relative to the window median and its darkness (1 - image along the ridge / image at the two cell centres).
- Family members (DINOv3, ConvNeXt, DINOv2) mixed into the HEX base on folds 0 and 4: -0.0006, left out (not worth a
  deployment).

Ridge evidence from other detector graphs on top (pre47, pre48: no fold picks it; frozen pre46, and the mean over 12 detector
graphs: f04 +0.001) adds nothing more: two ridge steps are kept, the direction is closed.

Dropped (HEX on folds 0+4 rose): multi-detector graph network (3 detectors, all three losses), CD/CV predictions as features, quadratic term, MAPE-loss linear fit on uncertainty features,
detector legibility counts, all prior features, all 77 consensus-point features, 49 legible-window statistics, richer image
statistics on top of the 4, vignetting subset, MAPE-loss image fit.

Earlier facts:
- Fusing detectors barely improves points on the 25 marker frames (miss 0.20 vs 0.22, false 0.17 vs 0.16). Detector errors
  are shared.
- Best-epoch selection flatters HEX by 0.017 only: it cannot explain submission 1's +0.40 test gap.
- HEX MAPE is steep in a global scale: 1 % costs +0.035, 2 % +0.14, 3 % +0.31.
- HEX bias by imaging year runs from -1.3 % (2021) to +0.3 % (2023). Year is not known at test time; image statistics stand
  in for it.

Graph networks on GPU, each 5 seeds x 5 folds, read half-and-half with the best pipeline's graph part (folds 0+4 / five):
control -0.0002 / +0.004; 4 image statistics in the head -0.016 / -0.023 (no CV gain); 25 statistics +0.003 / -0.009;
ridge node features -0.004 / -0.007. On top of the chain none adds: the chain's ridges already carry the same signals.
The chain has converged: second passes of every step and hex6 weights down to 0 change nothing.

HEX looks close to its ceiling. On the 25 expert-marker frames, marker hex6 gives r 0.84 with the label and hybrid B alone
already 0.81; the label is read from ~150 cells (sampling SE ~7.5 %) and residual log SD is 0.123.

Segmentation (zero-shot Cellpose-SAM, local `cpsam_models/cpsam_v2`, no training; `source/seg_hex.py`, `source/seg_run.py`):
- Loose thresholds (flow 0.8, cellprob -2) + CLAHE make 85-90 % of frames usable (default 53 %). Full run on all frames on
  GPU 1, masks kept (`seg_loose/`).
- On the first 1024 frames, mask hex6 r 0.47 with HEX (Voronoi hex6 the same) and r -0.06 with the chain residual. A ridge
  fit on segmented rows only: -0.01 to -0.03 on those rows. Region (centre, best window, top-3 windows) and neighbour radius
  (2, 3, 5 px) change nothing.
- On the 25 marker frames, mask hex6 inside the technician's box r 0.77 vs 0.73 for the whole frame; predicted boxes
  0.75-0.79 (IoU 0.42-0.54). The region is not the limit; cell-level error is.
- All frames done: the chain keeps `segm` only (status table). Centroid-Delaunay neighbour counts from the same masks (`segv`,
  `source/seg_vor.py`): -0.0003 on folds 0+4, +0.0003 on five folds, left out of the main chain. Cell graphs from the masks (`source/mask_graph.py`)
  feed a graph-network job on CPU (`mask_hex_s5c`).

Closed today: stretch per label source (ID format or the image classifier, AUC 0.999): +0.006 / +0.002 to +0.005.

## From-scratch combination search (PI via the coordinator, 2026-11-11 09:50-10:05 EST, CPU only)

Question: does a combination chosen without the historical order beat the 10-step chain (9.73258 five folds, 9.86437 folds 0+4)? Everything below is nested: every choice for outer fold f (step, option, share, ridge alpha, base) is made on the other four folds; the value is the five-fold OOF of the per-fold candidates through the usual per-fold stretch.

| Method | Five folds | Folds 0+4 | Notes |
|---|---:|---:|---|
| Current 10-step chain (steps decided on folds 0+4 / the audit rule) | 9.73258 | 9.86437 | step inclusion partly chosen on reported rows |
| Same chain, historical order, step kept per fold only if it lowers the other folds' HEX | 9.74255 | | nested estimate of the chain procedure; folds 2 and 4 skip `graphens`, `ridgeev_fall16` skipped by all |
| (1) Forward selection from the best pipeline, whole pool every round (`source/combo_search.py`) | 9.74827 | 9.87187 | stops after 3-5 steps; picks per fold: imgstats/uncridge, segm, attnp5/mtcnn5, segb, ... |
| (1) Forward selection from the ap1 base | 9.77078 | 9.88371 | stops after 3 steps |
| (2) Joint nested ridge, all corrections at once, deployable columns, ap1 base (`source/joint_fit.py`) | **9.72621** | **9.85123** | -0.0064 vs the chain; base chosen nested (ap1 on all five folds); alpha 10 / 1 / 1 / 1 / 1 |
| (2) Same, best-pipeline base | 9.73322 | 9.85352 | |
| (2) Joint ridge plus the backbone attention-pool DINOv3-B (no full-data build) | **9.71882** | **9.84826** | -0.0138; needs an attention-pool full-data build |
| (2) Joint ridge without the segmentation columns | 9.75186 | | segmentation carries the most |

- Pool for (1): 55 steps with five-fold OOF values, including dropped and closed ones (gnn:hex variants, segvig, single-seed and seed-mean cvw3, the 9 other ridge-evidence detector sets, image, prior, fuse and legibility groups, cvdino, attention-pool, mtcnn5). 3 could not run (cnn5: missing fold files; two context gnn variants: missing folds).
- Columns of the joint ridge (38): A/P1/h6/have blend (4), BEST and GRAPH HEX (2), graph ensemble s10x3 (1), HEX-only graph net pre46 (1), image statistics (4), prior cm1.0_h6c (1), ridge evidence k3 (5) and fall16 (1), segmentation group bh + usable flag (16), cvw3 seed mean (1), mtcnn5 (1), plus log base. Coefficient signs agree across the five outer folds on 29 of 38 columns.
- Every column of the deployable variant has a full-data build (cvw3 s46 + s47, CPSAM, graph nets, detectors). The attention-pool variant needs the backbone lane's full-data model.
- **Correction (10:00 EST, itself superseded at 10:10 EST, see below): the joint ridge gain looked not robust.** It depends on the base convention for the training rows inside each outer fold. With the chain convention (fold f's BEST on all rows, in-sample on the training rows, as every chain step and the 9.73258 are computed) it reads 9.72621 / 9.85123; with the deploy-like convention (OOF BEST on the training rows, as test images will look) it reads 9.74136 / 9.86929, worse than the chain. Nothing else differs (`source/joint_deploy.py`, nested parity 2.2e-16; side files `preds/deploy_params_joint.json`, `preds/chain_joint.npz`). Conclusion: no from-scratch combination beats the 10-step chain robustly; the honest nested numbers (forward selection 9.748, nested-keep chain 9.743, joint ridge 9.726-9.741) suggest the chain's 9.73258 carries about 0.01 of selection. Recommendation: keep the chain (coordinator decides).
- **Resolution (10:10 EST) and strict nesting.** The 10:00 correction compared the joint ridge under the deploy-like convention (9.74136) with the chain under the chain convention; the chain's own deploy fit (`chain_deploy.py`) also uses OOF inputs while its OOF number uses the chain convention, so like for like the joint ridge wins. Strict nesting (`source/strict_nest.py`: the whole procedure rerun with the scored fold's labels hidden): best pipeline 9.80262; chain procedure **9.74961**; fixed 10-step composition 9.74062; joint ridge **9.72662**; joint + attention pool 9.71977. Constants in the chain: shares 0.2 fixed in advance; option picks and ridge alphas chosen per fold on the other folds, but those folds' candidates were tuned with the scored fold's labels (second order); per-fold stretch on the other folds; step inclusion decided on folds 0+4 and the seed audit on all folds (direct). The best pipeline's 9.8026 uses the same per-fold stretch (identical under strict); its fixed constants (h6 blend 0.28, f6 map -0.2084 + 0.5989 f6) were set upstream on all rows and are shared by every number here. Coordinator adopted the joint ridge (FREEZE_CANDIDATE.md).
- Files: `preds/combo_forward*.json`, `preds/combo_joint3_{deploy,deploy_attnp}.json` and `_nestedbase.npz` (OOF candidates).

## Synthetic zero-shot columns (backbone lane, read 2026-11-11 12:15 EST)

Three label-free columns (log CD, log area CV, log HEX of a DINOv3-B pretrained only on synthetic specular frames, zero-shot on all frames; `lane_backbone_20261108/data/synth_zeroshot_f0pre_20261111.npz`) added to the 39-column joint ridge, strict five-fold read (`source/strict_joint_extra.py`): 9.72086 / 9.85013 against 9.71977 / 9.84826, **+0.00109**; per fold +0.0053 / -0.0033 / +0.0013 / +0.0036 / -0.0015. Closed.

## Synthetic-pretrained member, folds 0+4 (backbone lane, read 2026-11-11 13:50 EST)

The synthetic-pretrained DINOv3-B fine-tune (`lane_backbone_20261108/runs/synthft3_f{0,4}_dinov3b16_s46/raw`, epochs 8-12) entered the 39-column joint ridge as its log HEX deviation from the A/P1 blend on folds 0 and 4 and 0 elsewhere (`source/strict_part_member.py`; no availability flag, which would act as a fold indicator). Strict: 9.72078 / 9.84848 against 9.71977 / 9.84826 (+0.00101 / +0.00022); standardized weight -0.0016 to +0.0006 across outer folds. Weight about 0 for HEX. (Dry run of the same method with C-Mixup folds 0+4: +0.00108 / -0.00056.)

## Synthetic-pretrained recipe A (D-SYN-003, read 2026-11-12 02:05 EST)

Recipe A with the synthetic-pretrained s46 (final four-flip TTA) on folds 0 and 4 (the CD lane's `reads/recipeA_swapped_syn46_20261111.npz`) replaced A everywhere the HEX stack uses it (A, B = 0.5 (A + P1), H and the best-pipeline base), 39-column joint ridge rerun strict with plain A in the same run (`source/strict_swapA.py`): swap 9.72147 / 9.84491 against plain 9.71977 / 9.84826, **+0.0017 on five folds**, -0.0034 on folds 0+4; per fold -0.0011 / +0.0119 / +0.0023 / +0.0010 / -0.0055. Folds 1-3 move because their fits use the swapped folds 0 and 4 as training rows. No HEX gain on five folds (L2, L7).

## Seed audit of the freeze chain (coordinator, 2026-11-11 04:20-06:45 EST)

Rule fixed before the reads: a single-seed trained step gets a second seed on five folds (same recipe, only the seed changes) and is read with the mean over seeds; it stays if the chain with the seed mean is no worse than the chain without the step (five folds), else it drops; no seed is picked. Full tables in FREEZE_CANDIDATE.md, section "Seed audit".

- **cvw3** (s47 on GPUs 1-2, 36 min per fold): without 9.73308; s46 9.73579; s47 9.73020; mean **9.73222** -> stays. Folds 0+4 disagree (+0.0034). The s47 full-data build is done (`deploy/cvw3_f99_s47`, 2456 s, 06:53 EST); sanity: s47 r 0.978 with its OOF member, the 10-checkpoint deploy input r 0.980 with the OOF seed mean (`logs/cvw3_s47_sanity.json`). Fallback if it fails or misses 15:00 EST: the chain without cvw3 (9.73308), not single-seed cvw3.
- **gnn:hex_pre46_s5** (detector-seed check with pre47 graphs, 27 min on GPU 2): without 9.73258; pre46 9.73222; pre47 9.73258 (nested pick none); mean 9.73261 -> drops (by 3e-5). Folds 0+4 disagree (-0.0021).
- **mtcnn5** (CPSAM): a fold takes 8.6 h, so only a two-fold check ran (s47 folds 0 and 4; folds 1 and 2 were stopped when their GPUs went to the backbone lane). Read inside the 39-column joint ridge (D-HEX-004), seed 47 or the seed mean on folds 0+4 rows, seed 46 elsewhere, nested, same run: no column 9.71991 / 9.84918; seed 46 9.71882 / 9.84826; seed 47 9.71846 / 9.84785; seed mean 9.71827 / 9.84692. The column's small gain (-0.0011 to -0.0016 five folds) holds with the second seed; standardized weight 0.009-0.010 for both seeds. Under 0.005 (L7), so nothing changes; deploy keeps the one full-data seed.
- **New freeze chain**, 10 steps: ap1 h6w graphens imgstats prior ridgeev ridgeev_fall16 segm cvw3m_fix mtcnn5_fix = **9.73258 / 9.86437** (was 9.7358 / 9.8651). `preds/chain_seedaudit10.npz`, `preds/deploy_params.json` (11-step version archived), `hex_apply.py oof` parity 8.9e-16. Cell-route refit on it: HEX 9.72623 five folds / 9.8554 folds 0+4 (was 9.7283 on the 11-step chain).
- **Leave-one-out** on the 10-step chain: every step but `ridgeev` costs HEX when removed on five folds (`ridgeev` -0.00046 / -0.00227; then `graphens` turns -0.00108 on five folds). Coordinator: information only, no cascade (moves of about 0.001, choosing on reported rows).
- How it was run: `source/seed_audit_queue.sh`, `source/gnn_pre47_after.sh`, `source/mtcps_seed2.sh`, `source/chain_loo.sh`; chain reads with `CHAIN_FORCE=1` (every step kept; `RESULTS.tsv` rows marked "kept (forced)" are these audit reads).

## Vigueras-Guillen edge network (experiment, started 2026-11-10 22:00 EST)

Assigned by the coordinator (KG `idea-vigueras-guillen-weights`, never tried). It is the only member whose supervision is hand-traced cell edges from another specular dataset. Status: **experiment only**. Whether the challenge rules allow public pretrained weights in the final container is unverified and is the PI's call at adoption time.

- **Weights and source.** TVST 2020 CNN-Edge and CNN-ROI (`weights_edges_epoch150.h5`, `weights_ROI_epoch150.h5`, sha256 a32f14e2... and d46b26f6..., see `datasets/external_20261106/vigueras_tvst_weights/SHA256SUMS`).
  - Code: https://github.com/jpviguerasguillen/corneal_endothelium_TVST, commit 8d49345f.
  - Weights: https://drive.google.com/drive/folders/17xVFy7UfnHH66PfN5zlxv5swHwc-Ecdx
  - Paper: Vigueras-Guillen et al., TVST 2020; 9(2):49, https://doi.org/10.1167/tvst.9.2.49 (PMC7445361).
- **Licence.** The repository LICENSE is "GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007" (full text in `vigueras_tvst_weights/src/LICENSE`). The weights carry no separate terms.
- **Why this pair.** It is the only Vigueras network whose port can be checked against the authors' saved outputs: one example image with its CNN-Edge and CNN-ROI outputs. The 2022 fNLA nets have no reference output, and the input order of their body net is undocumented.
- **Safe read** (coordinator conditions, 2026-11-10 EST).
  - h5py 3.11.0 from PyPI, installed with `--no-deps --target pylib/` only. Wheel sha256 aa6ae84a14103e8dc19266ef4c3e5d7c00b68f21d07f2966f0ca7bdb6c2761fb, checked against PyPI. Not in the shared venv and not in any deploy image.
  - `source/vig_convert.py` runs under python -I. It walks the HDF5 tree link by link, refuses any soft or external link, external storage or non-float dataset, and reads no attributes.
  - Output: tensors-only `deploy/vigueras/tvst_{edge,roi}.pt` (7,129,204 parameters each, matching the authors' 7.1M), loaded with `torch.load(weights_only=True)`.
  - Nothing from the authors' repository is imported or run. The architecture is my own PyTorch port (`source/vig_net.py`).
- **Port check against the authors' example** (`logs/vig_verify.json`, `source/vig_diag*.py`).
  - CNN-Edge vs their NN_Edges.tif: max abs 159/255, mean 4.1/255, corr 0.982.
  - CNN-ROI on their saved edge map vs NN_ROI.tif: max abs 208/255, corr 0.992.
  - Not an exact match. Every alternative convention collapses the match: padding 2-before 0.04, flipped kernels 0.40, transposed-conv crop offset 0 or 2 → 0.45 / 0.50. BN statistics are identical across variants, and the residual is spread evenly over the image.
  - Reading: the port is right, and the example was made with different weights (for example a cross-validation fold model). So the port matches the authors' example at corr 0.982 (edge) / 0.992 (ROI), not exactly; the convention tests pin the port and the remaining gap is attributed to the weights. Coordinator decision (2026-11-10, about 22:20 EST): run with the released weights, since the convention tests do the job of the exact-match condition (catching a wrong port) and the stack read decides the rest. Before the full run: a 200-frame sanity (random, seed 0, CNNs on the CPU so GPU 1 stays with the full-data CPSAM build), cell counts against CPSAM on the same frames; stop and close if clearly broken (r < 0.5 or edges do not close into cells). **Sanity result (2026-11-10 23:03 EST, logs/vig_sanity.json): passed.** On 184 frames with both routes finite, Vigueras vs CPSAM: interior cells r 0.888 (median ratio 0.94), all kept cells r 0.785 (ratio 0.76; Vigueras keeps only ROI cells, median ROI fraction 0.35), density r 0.916 (ratio 1.07), h6 r 0.62 (median 0.45 vs 0.50), area CV r 0.30 (0.38 vs 0.29). Edges close into cells: median 471 interior cells, 170 of 200 frames have >= 50 (CPSAM 180). Median fc 0.056 /px (cell width about 18 px at 1.04 um/px). Full run armed on GPU 1 after the CPSAM build exits.
- **Planned read.**
  - Run: frames resized to the authors' pixel size (720 x 960, 1.042 um/px; theirs is 1.038), intensity / 255, then CNN-Edge, then CNN-ROI (`source/vig_infer.py`).
  - Cells: Python re-implementation of their MATLAB postprocessing (frequency-scaled smoothing, watershed, their correction step), with a cell kept when 75 % of it is in the ROI (`source/vig_post.py`).
  - Features: the CPSAM route's `feats()`, a verbatim copy checked against a stored CPSAM chunk (max diff 1.1e-4, float32 rounding), plus fc and the ROI fraction.
  - Stack read: five-fold HEX through the `segm` recipe on the current chain (9.7358 / cell-route 9.7283), plus residual correlation. CV features go to the CV lane. Weight 0 closes it. Box 12 GPU-h.
- **Result (2026-11-11 02:30 EST): closed (weight about 0).** Full run: GPU inference 8954 frames in 35 min on GPU 1 (after the CPSAM build exited), CPU postprocessing 4 workers 54 min -> `preds/seg_vig_all.npz` (17 CPSAM-route features + vig_fc + vig_roi_frac; 7136 usable rows, CPSAM 7704). Chain read (main 11 steps + segvig, `logs/chain_segvig.log`): folds 0+4 9.8703 (+0.0052), five folds 9.7361 (+0.0003), per fold +0.013 / -0.011 / +0.010 / -0.008 / -0.002, dropped. Residuals (`logs/vig_resid.json`): the step's nested correction has sd 0.004 (chain residual sd 0.119) and correlates 0.013 with the chain residual (per fold -0.006 to 0.087); a standalone HEX estimate from the Vigueras features alone has residual r 0.904 with the chain residual and r 0.954 with the same estimate from CPSAM features; standalone MAPE 10.32 vs CPSAM 10.07 vs the chain 9.42 on the same rows. Reading: on our frames the Vigueras cells carry the same HEX information as the CPSAM cells, slightly noisier, and nothing the chain lacks. CV features handed to the CV lane; its read (nested ridge, alpha 100, 7136 rows): +0.0196 on its candidate stack, every fold worse; Vigueras area CV tracks the CV label (r 0.41) but not the stack residual (r 0.02), so the direction is closed for CV too. GPU use about 0.6 GPU-h of the 12 GPU-h box.
- **Literature.** No published test of these weights on another specular microscope was found (search 2026-11-11). KG links: `viguerasguillen2020dsaek`, `viguerasguillen2022feedbackattn`, `viguerasguillen2019regression`. This direction overlaps `dec-014` (mask routes closed for CV/HEX); it runs because the coordinator assigned it as a different input (an in-domain edge model).

## GPU

Coordinator allowed small graph jobs on GPU 1 next to CPSAM fold 4 (at most 2, about 4 GB each, CPSAM epoch under ~60
min). One unthrottled job still cost CPSAM 10.5 s per update against 5.8 alone (epoch 6: 78 min), so the trainers now
sleep after each update to a duty set in `duty_gpu1.txt`: at 0.25 with two jobs CPSAM runs 6.8 s per update (+17 %).
`source/queue_gpu1_v2.sh` runs `jobs_gpu1.tsv`; concurrency is read from `slots_gpu1.txt` (now 1, plus the segmentation run at duty 0.5:
CPSAM 7.2 s per update, ~57 min per epoch). `source/post_cpsam.sh` sets duty 1 and 2 slots when CPSAM's pid exits. A d64 5-layer graph network: 75 s per seed-fold
alone, 99 s with two jobs; about 40 min per 5 seeds x 5 folds. Order: control, image statistics in the network head
(`--ctxfile`, 4 and 25 statistics, HEX-only loss), ridge node features (14), ridge + appearance node features (18), HEX-only
loss, multi-detector network, other detector graphs, window variants.
CPU screen (d32, 1 seed): the multi-detector network is no better than its control (stack +0.0002 vs +0.0015).
