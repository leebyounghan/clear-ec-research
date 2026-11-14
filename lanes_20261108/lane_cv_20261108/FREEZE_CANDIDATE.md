# CV lane freeze candidate (written 2026-11-09 11:00 EST; updated 2026-11-11 03:15 EST: the pre-registered 3-seed rule selected the 6-component stack, CV 10.1101)

## Result

| Item | Value |
|---|---|
| Five-fold OOF CV | **10.1101** |
| Baseline | best pipeline `experiments/meta_ceiling_20261103/source/best_pipeline_20261106.py`, REF_BEST CV **10.1512** |
| Change | **-0.0411**; per fold -0.103 / +0.055 / -0.059 / -0.063 / -0.035. Decided 2026-11-11 by the coordinator's pre-registered rule: the stack with the order-loss DINOv3 as the mean of seeds 46, 47, 49 reads -0.0369 (CV 10.1143), so it does not beat this stack. Seed 46 alone (-0.0460) was ruled out as a seed pick on reported rows. Earlier candidate: -0.0370 (without the order graph quartet) |
| CD, HEX | unchanged (only the CV column is corrected) |
| Strictly nested value | **10.1160** (-0.0352). The same 6 components refit with the inner stretch also excluding the outer fold, from `source/cv_search_20261111.py` on 11-11. cv_stack5's inner stretch sees the outer fold, so 10.1101 is about 0.006 optimistic. A from-scratch search over 25 components under strict nesting does not beat this stack: grid 10.1176, forward selection 10.1324. |
| OOF file | `experiments/lane_cv_20261108/out/cv_final_20261110.npz` (= `out/stack5_plain_order_2comp.npz`), with ids / idx / fold in `deploy/cv_final_oof.npz`: `P` = N x 3 predictions in best-pipeline row order (CD and HEX columns equal REF_BEST), `names`, `w0..w4` = nested weights per outer fold, `w_all` = deploy weights |
| Reproduction | `DINO_SEEDS=46,47,49 python source/cv_stack5_20261108.py --save freeze_cv_repro2 dino fz:spectrum fz:cov_sqrt fz:autocorr graph:tta_s46_tta4_5f+tta_s47_tta4_5f+tta_s48_tta4_5f+tta_seeds3_tta4_5f graph:gord_s46_tta4_5f+gord_s47_tta4_5f+gord_s48_tta4_5f+gord_seeds3_tta4_5f` (run from the lane folder; see the check at the end) |

One recipe on every fold: the same members, seeds and settings on all five folds.

## Formula (per image, CV column only)

```
log CV_pre = log H_cv                                   # hybrid B CV (hex6 changes only HEX)
           + BEST_cv - GRAPH_cv                         # best pipeline's simulation-embedding ridge, without its graph part
           + w_g  * QUARTET_cv                          # TTA graph quartet replaces the shipped graph
           + w_o  * (ORDER_QUARTET_cv - GRAPH_cv)       # order-loss graph quartet (member 5), a second graph component
           + w_d  * (log DINO_cv - log A_cv)            # DINOv3 B/16 member vs recipe A (TTA scalar)
           + w_s  * (lg_spectrum - lg_std) + w_c * (lg_cov_sqrt - lg_std) + w_a * (lg_autocorr - lg_std)   # frozen heads
CV = stretch(exp(log CV_pre))                           # best pipeline's log-affine stretch (N.fit_stretch / apply_stretch), refit on all rows
```

(The stacker writes each graph term as w x (members' mean CV - GRAPH_cv) added to BEST_cv. With w_g = 1 the first equals the line above.)

**Deploy weights (fit on all five folds):** w_d = 0.125, w_s = 0.25, w_c = 0.375, w_a = 0.25, w_g = 1.0, w_o = 0.25. Nested weights per outer fold are in the OOF file.

## Members

### 1. DINOv3 B/16 member (seeds 46, 47, 49; mean of log CV)

| Seed | Folds 0, 4 | Folds 1, 2, 3 |
|---|---|---|
| 46 | `experiments/lane_gpu3_20261106/runs/dinov3_b16_f{0,4}_s46` | `experiments/lane_cd_20261108/arms/dinov3_b16_f{1,2,3}_s46` |
| 47 | `experiments/lane_gpu2_20261106/runs/dinov3_b16_f{0,4}_s47` | `experiments/lane_cd_20261108/arms/dinov3_b16_f{1,2,3}_s47` |
| 49 | `experiments/lane_cv_20261108/arms/dinov3_b16_f{0..4}_s49` | same |

- Prediction per run: `runs/fold{f}/dinov3_b16/predictions_epoch_{16..20}.npz`. The member is exp(mean of the log predictions over epochs 16-20), with no TTA. Seeds are averaged in log space.
- Recipe: lane GPU 3 family runner (`experiments/lane_gpu3_20261106/source/family_recipe_20261106.py`; identical copies in `lane_cd_20261108/source/family_cd_20261108.py` and `lane_cv_20261108/source/family_cv_20261108.py`, run without CD_LOSS, CROP_LW or AUG_BUNDLE set). It runs 5 LeJEPA SSL epochs on the fold's training images, then 20 regression epochs (std pooling, schedule10, encoder LR 1e-5, head LR 1e-3, recipe loss, flips).
- Full-data training: the family runner has no all-rows split wired. Point `SPLITS` at an all-rows train split, e.g. `experiments/fulldata_recipeA_20261028/fulldata_regression_split.npz`, then run `python <runner> all --family dinov3_b16 --fold <id> --seed {46,47,49} --root <dir>`. Measured time per seed on one H200: SSL 36 min (fold 1, shared GPU) plus 55-60 min regression on a fold (7151 rows) with the GPU to itself. That gives about 1.8 h per seed on full data, about 5.5 h for three seeds in sequence, under 2 h in parallel on separate GPUs.
- Inference: the runner's validation path (std-pooled tokens, then the head); input is the 972 x 1296 frame (CLAHE plus 1-99 percentile scaling, as in training).

### 2. Frozen token-statistic heads (spectrum, cov_sqrt, autocorr)

- Encoder: recipe A seed-46 epoch-20 encoder of each fold (OOF): `experiments/lane_gpu2_20261106/source/token_stats_20261106.py` `ENC` dict. For deployment, use the full-data recipe A seed-46 encoder (`experiments/fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt`).
- Features ("own" read: each fold's encoder on all rows): folds 0 and 4 in `experiments/lane_gpu2_20261106/frozen_own_f{0,4}/`, folds 1-3 in `experiments/lane_cv_20261108/frozen_own_f{1,2,3}/` (blocks std, spectrum, cov, cov_sqrt, autocorr; PCA basis `experiments/lane_gpu2_20261106/frozen_s46/pca.npz`).
- Heads: `experiments/lane_gpu2_20261106/source/frozen_read_20261106.py` BlockHead on [std + block], recipe loss, 3 head seeds, 20 epochs, mean of epochs 16-20 log predictions. Arms: `std`, `spectrum`, `cov_sqrt`, `autocorr`.
- OOF head predictions: folds 0 and 4 in `experiments/lane_gpu2_20261106/reads/own/{arm}_f{0,4}.npz`, folds 1-3 in `experiments/lane_cv_20261108/reads/own/{arm}_f{1,2,3}.npz` (`rows`, `lg` = N x 3 log predictions).
- Full-data training: extract the blocks with the full-data encoder on all rows (`source/token_stats_cv_20261108.py own` pattern, about 35 min for 8954 frames at 4 fps on a shared GPU), then fit the four heads on all labelled rows (minutes).
- Inference: encoder forward on the frame, block statistics, four small MLP heads.

### 3. TTA graph quartet (replaces the shipped graph's CV)

- Detector peaks: the three pre-only detectors (seeds 46, 47, 48; `experiments/green_track_20261028/external_srep/{fold{f}, s47/fold{f}, s48/fold{f}}/sg2_pre.pt`, each initialised from fold f's recipe A encoder, pre-trained on the external srep set, never fine-tuned on challenge images). Four exact flips; heatmaps averaged; peaks by `SG.peaks2(map, support, 0.2)` (NMS 12 px). Sets: `s46_tta4`, `s47_tta4`, `s48_tta4` (each detector's flip mean) and `seeds3_tta4` (mean over 3 detectors x 4 flips). Script: `source/tta_infer_20261108.py`. Peaks in `out/tta_peaks/<set>/fold{f}/fold{f}_sg2_peaks.npz`.
- Graphs: `source/graph_prep_sp_20261108.py tta_<set>.npz out/tta_peaks/<set> 3 380 300 0.35 0` (3 legible 380 x 300 windows from peaks >= 0.5, nodes >= 0.35, Voronoi edges; node features as the shipped graph).
- Graph network: `source/graph_train_cv_20261108.py` (copy of `cell_graph_sweep3_20261105.py`) with `--d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx`. Five-fold OOF files: `graph_runs/tta_{s46,s47,s48}_tta4_5f.npz`, `graph_runs/tta_seeds3_tta4_5f.npz` (key f{f} = N x 3 log residual corrections of fold f's model). QUARTET_cv = mean of the four members' CV column.
- Full-data training: (a) three full-data pre-only detectors, initialised from the full-data recipe A encoder and pre-trained on srep with 2000 updates (`experiments/green_track_20261028/source/green_external_srep_20261030.py pretrain`, about 5 min each); (b) TTA peaks on all frames (12 detector passes per image, about 2.5 h for 8954 frames on a free H200); (c) graphs (CPU, about 1 min per set with 4 workers); (d) graph network on all labelled rows, 3 seeds x 4 sets. The trainer is per fold: run it with the train mask = all good rows. That is about 25 s per epoch-set, under 1 h in all.
- Inference: 12 detector passes, then peaks, the 3 windows, the graph, and 4 graph networks x 3 seeds.
- **Detector pair (decided 2026-11-10).** The OOF quartets used the fold detectors above, and every full-data graph net (plain and order quartets, the new GRAPH fits) was trained on graphs from fold-detector peaks, so the deploy pair only matters at test time. Deployment uses the green-track pair `green_track_20261028/deploy/detectors/sg2pre_s{46,47,48}.pt` (10-30; md5 06c90eee7e8c60cc9ac0ca814dc2e2bf, c84866ca467228737757cadcfdb0c728, 4733d1fc0855211fb20512cd15221482), the same recipe as the fold detectors. Do not use `lane_cellroute_20261109/deploy/detectors/sg2pre_s{47,48}.pt` (11-09; md5 ef64623a..., 7d5728f0...; same recipe, different weights) for CV. The cell-route lane agreed to switch to the green-track pair.

### 5. Order-loss graph quartet (added 2026-11-10)

- Same four graph sets, detectors, graphs, network and settings as member 3, trained with an extra loss term: 1.0 x noise-weighted pairwise BCE on the CV column within each batch of 64 graphs (p_ij = Phi(d log y / (sqrt 2 x 0.065)), logits d log y_hat / (sqrt 2 x 0.065)); KG idea-cv-noise-weighted-order-loss, lee2026stochorder arXiv:2607.08103. Script: `source/graph_order_cv_20261110.py --order 1.0` (copy of `graph_train_cv_20261108.py`; `GRAPH_DEV=cpu` runs on CPU, `--seed0 K` splits seeds over processes).
- Five-fold OOF files: `graph_runs/gord_{s46,s47,s48,seeds3}_tta4_5f.npz` (mean of 3 seeds per fold; per-seed files `gord_<set>_tta4_f<f>_s<k>.npz`). Fold 0 of seeds3 ran on GPU, the rest on the CPU pod; same recipe.
- ORDER_QUARTET_cv = mean of the four members' CV column.
- Alone, five folds: order quartet in place of the plain quartet -0.0265; plain and order averaged -0.0251; as its own component (this candidate) -0.0411.

### 6. Order-loss DINOv3 B/16 (NOT in the candidate since 2026-11-11; kept for the record)

- The DINOv3 member's recipe (member 1) with `CD_LOSS=order`: recipe loss + 1.0 x noise-weighted pairwise BCE on CV in each batch (same order term as member 5). Seed 46, SSL reused from the seed-46 member runs. Script: `source/family_cv_20261108.py all --family dinov3_b16 --fold F --seed 46 --ssl-from <member-1 s46 SSL>` with `CD_LOSS=order` (`source/qR5.sh`).
- OOF: `arms/dinov3_order_f{0..4}_s46/runs/fold{f}/dinov3_b16/predictions_epoch_{16..20}.npz`; member = mean of log CV over epochs 16-20, single view. Alone, five folds: -0.0190 (the plain s46 member: -0.0093). As its own component: -0.0460 (this candidate); inside the DINOv3 mean instead: -0.0360.
- Full data: `arms/dinov3_order_f99_s{46,47,49}/runs/fold99/dinov3_b16/epoch_{16..20}.pt` (s46 4102 s on GPU 2; s47 5072 s and s49 4868 s on GPU 3).
- Seeds: s47 alone -0.0214 and s49 alone -0.0173 as members. In the stack, the s46+s47 mean reads -0.0404 and the 3-seed mean -0.0369, both worse than -0.0411. The one-seed gain did not hold, so the pre-registered rule dropped it.

### 4. Shared parts from the best pipeline (unchanged in the OOF; deploy fits owned by the CV lane from 2026-11-10)

H (hybrid B plus hex6), BEST (`sweep/sim_pred/sim2_l8_ongraph_nested.npz`), GRAPH (mean of `graph_sweep/d64_l5_noctx_s10`, `d64_l5_pre47_s3`, `d64_l5_pre48_s3`), stretch: all from `best_pipeline_20261106.py`. In the OOF they existed only as nested per-fold arrays. Deploy versions (all rows, same recipes) are built by `source/qC4.sh` (GRAPH nets, CPU pod) and `source/cv_deploy_fit_20261110.py` (ridge, scalars, stretch); see "Deploy" below.

## Deploy (CV apply, 2026-11-10)

- Entry point: `source/cv_apply_20261110.py` (numpy only): `P = load_params()`; `cv = apply(P, m)` with per-image member values (docstring lists them); `ridge(P, embs, 'CV'|'HEX')` gives the simulation-embedding ridge alone (the HEX lane uses the HEX column). No labels, fold ids or OOF files at inference.
- Check: `python source/cv_apply_20261110.py check` applies each fold's parameters to the OOF member values and reproduces the stored OOF CV column (max |diff| 2.8e-16, CV 10.1101).
- OOF tables: `deploy/cv_final_oof_20261110.npz` (P, cv, ids, idx, fold, weights; 8954 rows, ids unique) and `deploy/cv_inputs_oof.npz` (each apply input per row from the model that did not see it, with per-fold weights, recipe A scalar and stretch; `apply_fold(Z, f)` reproduces the OOF CV, max |diff| 2.2e-16).
- Deploy parameters for this candidate: `deploy/cv_deploy_params.json` (md5 3b437500) and `deploy/cv_ridge_params.npz`. They were built as `deploy_fallback_10.1101/` with `CV_OOF=cv_final_20261110 CV_DEPLOY_DIR=deploy_fallback_10.1101 cv_deploy_fit_20261110.py ids|inputs|fit` and copied into `deploy/` on 11-11. Its CV stretch is (0.975, 0.98029, -0.98112); the in-sample sanity read is 10.0265. The 7-component params are in `deploy/_superseded/order7comp_20261111/`. Same fit procedure as below:
  - Recipe A two-seed scalars on all rows: CD 0.99141, CV 0.98911, HEX 0.99159. Phase-1 scalars: the fixed `C.SCALE`.
  - Simulation-embedding ridge (BEST - GRAPH), CV and HEX columns. Strength by leave-one-fold-out over the five folds: CV 30000, HEX 3000. Fit on all good rows against the residual of the all-rows GRAPH nets, which is in-sample, as in the nested recipe.
  - Stack weights `w_all` (above).
  - CV stretch (g, c, mu) = (0.975, 0.98136, -0.98117). It is fit on all good rows of the deploy pre-stretch prediction: graph parts in-sample from the all-rows nets, DINOv3 and frozen heads out-of-fold, the way the nested stretch saw them. For comparison, a fit on the OOF pre-stretch values gives c = 0.97858.
  - Sanity: the deploy parameters on the training rows give CV 10.022. This is optimistic because the graph parts are in-sample; it is not a score.
- All-rows GRAPH ensemble (the best pipeline's `GRAPH`, now deployable): `graph_models/gfull_{noctx,pre47,pre48}_s<k>_s<k>_e{21..25}.pt`, seeds 0-9 / 0-2 / 0-2.
  - Same recipe as `graph_sweep/d64_l5_{noctx_s10,pre47_s3,pre48_s3}`: d64, 5 layers, 25 epochs, no ctx, the identical 9-feature network, verified against the sweep script.
  - Trained on the CPU pod (`source/qC4.sh`), about 1.9 h per seed on 1 thread.
  - In-sample predictions: `gfull_*_fullfit_pred.npz`.
- Per-image parts: `source/cv_image_parts_20261110.py`. It has graphs from peaks, graph-net forward, simulation embeddings, token blocks and frozen heads, and imports nothing that reads labels. Parity on stored training rows: rebuilt graphs identical (feature difference 0); per-image net forward vs the training-time batched predictions 4e-9; embeddings within float16 storage precision (2e-3); cov_sqrt identical.

## Not in the candidate (tested, no gain on top)

Order-loss graph quartet in place of the plain one (-0.0265) or averaged with it (-0.0251); ViT-H+/16 and the recipe A window member on top (weight 0 / worse, folds 0+4); order loss on frozen heads; DINOv3 seed 50 (four seeds: -0.031), DINOv3 flip-TTA predictions, CV-weighted DINOv3, ViT-L/16, frozen heads on DINOv3 tokens, threshold-0.25 / 40- and 50-epoch / 5-detector TTA graphs, segmentation features, Konan weighted-bisector CV from detector peaks. Details in `RESULTS.tsv`.

## Reproduction check

Candidate (-0.0411): see 'Previous candidate (-0.0411)' below for its reproduction, and `python source/cv_apply_20261110.py check` (default stack file `cv_final_20261110`), which reproduces its OOF CV column to 2.8e-16 (rerun 11-11). Order-DINOv3 7-component version (-0.0460, not the candidate): `logs/freeze_repro3.log`. Verified 2026-11-10 12:50 EST: `NESTED STACK CV on folds 01234: 10.1052` with the same per-fold values; max |P difference| vs `out/cv_final_20261110b.npz` = 0.0; deploy weights equal. `python source/cv_apply_20261110.py check` (per-fold parameters on OOF members) reproduces the OOF CV column to 2.8e-16. Deploy files: md5 `cv_deploy_params.json` d6647d83, `cv_ridge_params.npz` e29fc85f.

Previous candidate (-0.0411): `logs/freeze_repro2.log` reruns the stack above. Verified 2026-11-10 06:50 EST: printed `NESTED STACK CV on folds 01234: 10.1101` (-0.0411, same per-fold values), deploy weights equal to `w_all` of `out/stack5_plain_order_2comp.npz`, max |P difference| = 0.0.

Previous candidate (-0.0370):

`logs/freeze_repro.log` reruns the stack from the stored member predictions with the seeds pinned (`DINO_SEEDS=46,47,49`). It must print `NESTED STACK CV on folds 01234: 10.1142` and deploy weights equal to `w_all` above, and `out/freeze_cv_repro.npz` must match `out/stack5_best_3seeds.npz` (`P`).

Verified 2026-11-09 11:10 EST: the rerun printed 10.1142 (-0.0370, same per-fold values), deploy weights equal to w_all, and max |P difference| = 0.0.

## Full-data builds (2026-11-09, GPU 2, measured)

| Part | Output | Command | Measured time |
|---|---|---|---|
| DINOv3 B/16 seed 46 | CD lane `arms/dinov3_b16_f99_s46b` (SSL `arms/dinov3_b16_f99_s46/ssl/last.pt`) | built by the CD lane | see CD lane |
| DINOv3 B/16 seeds 47, 49 | `arms/dinov3_b16_f99_s{47,49}/runs/fold99/dinov3_b16/epoch_{16..20}.pt` | `source/qF.sh`: `family_cv_20261108.py all --family dinov3_b16 --fold 99 --seed {47,49} --ssl-from <CD f99 SSL>` (fold 99 = `fulldata_regression_split.npz`, 8952 train rows, the 2 HEX-zero rows held; `safe_metrics` returns -1 for a column with no scorable rows) | s47: 5115 s (85 min, regression only, SSL reused; GPU shared with light jobs); s49: 6499 s (108 min, GPU shared with ViT-H+ SSL) |
| Frozen-head features | `frozen_own_f99/` (std, spectrum, cov, cov_sqrt, autocorr on all 8954 rows) | `token_stats_cv_20261108.py own 99` (encoder `fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt`) | 1602 s (GPU shared with a DINOv3 run) |
| Frozen heads | `frozen_heads_full/{std,spectrum,cov_sqrt,autocorr}_s{0,1,2}_e{16..20}.pt` (state dict, blocks, mu, sigma) | `source/frozen_full_heads_20261109.py` (all good rows; same head, loss, schedule as the OOF read) | 405 s |
| Pre-only detectors (TTA quartet input) | `green_track_20261028/deploy/detectors/sg2pre_s{46,47,48}.pt` (from the full-data recipe A encoder, 2000 srep updates) | already built (green track) | n/a |
| Order-loss DINOv3 s46 (member 6) | `arms/dinov3_order_f99_s46/runs/fold99/dinov3_b16/epoch_{16..20}.pt` | `source/qR6.sh`: `CD_LOSS=order family_cv_20261108.py all --family dinov3_b16 --fold 99 --seed 46 --ssl-from lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt` | 4102 s (GPU 2 alone) |
| All-rows GRAPH ensemble (best pipeline's GRAPH) | `graph_models/gfull_{noctx,pre47,pre48}_s<k>_s<k>_e{21..25}.pt` | `source/qC4.sh` (CPU pod) | about 1.9 h per seed, 1 thread, 16 in parallel |
| Order-loss graph networks (member 5) | `graph_models/gord_{s46,s47,s48,seeds3}_tta4_full_s{0,1,2}_s{0,1,2}_e{21..25}.pt` (+ `_fullfit_pred.npz`) | `source/qC1.sh`, `qC2.sh`, `qC3.sh` (CPU pod): `graph_order_cv_20261110.py gord_<set>_tta4_full_s<k> --graphs graphs/tta_<set>_tta4.npz --order 1.0 --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 1 --seed0 <k> --no-ctx --folds full` | about 1.45 h per seed on 1 CPU thread (12 fits in parallel, about 1.7 h wall); seeds 1 and 2 rerun 2026-11-10 after a --seed0 bug in the full branch (wrong ones in `graph_models/_archive_wrong_seed_20261110/`) |
| Graph networks (TTA quartet) | `graph_models/full_tta_{s46,s47,s48,seeds3}_tta4_s{0,1,2}_e{21..25}.pt` | `source/qG.sh`: `graph_train_cv_20261108.py full_<set> --graphs graphs/<set>.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds full` (all good rows) | 343 / 406 / 406 / 411 s (about 26 min for all four) |

Inference at deploy, per image: (1) 12 detector passes (3 deploy detectors x 4 flips), peaks per set (`SG.peaks2(.., 0.2)`, NMS 12), 3 legible windows, graph (`graph_prep_sp_20261108.graph`) -> 4 sets x 3 seeds x 5 epoch snapshots of the graph network, mean of the log-residual CV outputs, and the same for the order-loss nets (member 5); (2) full-data recipe A encoder -> token blocks -> 4 heads x 3 seeds x 5 epochs, mean log prediction per arm; (3) DINOv3 seeds 46/47/49 full-data, epochs 16-20 mean log prediction; (4) the formula above with the deploy weights. The graph nets were trained on graphs from the fold detectors' TTA peaks (the OOF inputs); at deploy the inputs come from the full-data detectors (same recipe, different encoder init).
