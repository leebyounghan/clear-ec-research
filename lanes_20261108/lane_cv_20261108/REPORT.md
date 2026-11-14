# CV lane (GPU 2), 2026-11-08

Goal: 5-fold OOF CV <= 9.99 (best pipeline 10.151), stretch 9.85, by the 11-11 18:00 EST freeze. Every try is one line in RESULTS.tsv.

## Where it stands (2026-11-08 02:45 EST)

- **Update 2026-11-10 08:15 EST: lane best CV 10.1101 vs 10.1512 (-0.0411; folds -0.103/+0.055/-0.059/-0.063/-0.035)**, one recipe on all five folds. It is the stack below plus the order-loss graph quartet as a second graph component (deploy weight 0.25). File: out/stack5_plain_order_2comp.npz; FREEZE_CANDIDATE.md updated and reproduced. The -0.037 line below is the previous candidate.
- Lane best, one recipe on all five folds (nested weights): CV 10.114 vs best pipeline 10.151 (-0.037; folds -0.096/+0.058/-0.069/-0.055/-0.023). Components: DINOv3 B/16 member (mean of seeds 46, 47, 49 on every fold), frozen token-statistic heads on the recipe A fold encoders (spectrum, cov_sqrt, autocorr), graph network on flip-TTA / 3-seed-averaged pre-only peaks (4 members). Deploy weights (fit on all five folds): dino 0.125, spectrum 0.25, cov_sqrt 0.375, autocorr 0.25, TTA graph 1.0 (replaces the shipped graph CV). File: out/stack5_best_3seeds.npz. The earlier -0.040 (out/stack5_dino_fz_tta.npz) used DINOv3 seed 47 on folds 0 and 4 only: a per-fold member difference, withdrawn.
- Fold 1 gets worse with every image component (+0.04 to +0.08); no fold-level offset in the components explains it.
- Dropped after reads: spacing-prior node selection (2 versions), 5 windows, box-size windows, window attention (3 versions), graphs from fine-tuned detectors, sim-initialised graph members, watershed boundaries, CV-weighted / CV-only frozen heads, frozen heads on DINOv3 tokens, CV re-weighting of the shipped graph members, CV from the pipeline HEX/CD information, recipe A vs Phase-1 base weight for CV.
- Running: DINOv3 B/16 with CV-weighted loss (CV x3), five folds; flip-TTA + 3-seed heatmap peaks -> graphs; pre-only detector seeds 49/50 -> graph members (five folds).

## Reading

- source/cv_compare_20261108.py: graph runs against the shipped graph on folds 0+4.
- source/cv_stack5_20261108.py: five-fold nested stack of CV corrections on top of the best pipeline (weights chosen on the other folds).

## Directions, continue or stop (2026-11-08 18:00 EST; next review 11-09 09:00 EST)

| Direction | Used | Decision |
|---|---|---|
| Graph variants on detector peaks (node selection, windows, attention, epochs) | about 1 GPU-day | **Stop** after the running 40-epoch and seed 49/50 TTA graphs land. Kept: TTA t0.35 quartet (in the lane best). |
| Frozen token-statistic heads (recipe A fold encoders) | done | Kept in the lane best; no more variants. |
| DINOv3 B/16 members (recipe loss, CD lane seeds) | read-only | Continue: add seed 47/48 folds as they land. CV-weighted loss: no gain, last fold finishing. |
| Cross-lane members: ViT-L/16 (backbone/CD lanes), CPSAM 512 multi-target (HEX lane) | read-only | Continue: read in the -0.040 stack when five folds exist (ViT-L tonight, CPSAM 11-10 morning). |
| Whole-frame graph on TTA nodes (more cells read) | new, under 2 GPU-h | One try, then stop. |


## Daily review 2026-11-09 09:00 EST (continue or stop)

| Direction | Decision |
|---|---|
| Graph variants on detector peaks | Stopped 11-08 evening. TTA quartet stays in the lane best. |
| Frozen token-statistic heads | Done, in the lane best. |
| DINOv3 B/16 seeds (label-trained image model, the one lever left per the root-cause read) | Continue: seed 50 training on GPU 2 (folds 0, 1, 4 done); add to the DINOv3 mean when all five folds land. |
| Cross-lane members | ViT-L/16 closed (five folds: alone -0.003, weight 0 in the stack). CPSAM 512 multi-target (HEX lane): read when its folds land. |
| Segmentation features (HEX lane) | Stopped: +0.011 on top of the stack (two runs). |

## Daily review 2026-11-10 09:00 EST (continue or stop)

| Direction | Used | Decision |
|---|---|---|
| Noise-weighted pairwise order loss on CV (label-noise direction; KG idea-cv-noise-weighted-order-loss, lee2026stochorder): v1 frozen heads | under 1 GPU-h | Stopped 11-09: no gain on folds 0+4. |
| Order loss v3: graph network quartet | about 0.1 GPU-h + about 105 CPU-h (CPU pod, coordinator-approved for the five-fold confirmation; about 75 runs of about 1.4 h, including the 8 full-data reruns after the seed bug) | **Kept, closed**: five folds as its own stack component -0.0041 (lane best -0.0411). In place of the plain quartet -0.0265, averaged -0.0251. Full-data nets built. No more variants. |
| Order loss v2: DINOv3 B/16 seed 46 | 2 GPU-h so far | **Continue to five folds** (folds 1-3 on GPU 2, done about 11:00 EST): folds 0+4 inside the DINOv3 mean -0.0074 on the lane best; as its own component 0. If five folds hold, add to FREEZE_CANDIDATE.md and build full data (about 1.5 h). |
| Cross-lane members on CV | read-only | ViT-H+/16: closed (weight 0 on both stacks; coordinator closed it). Recipe A window member (backbone lane): alone -0.0115 on folds 0+4, worse on top of the order-graph stack; read again at five folds (folds 2, 3 running on GPU 3 / GPU 0). |
| GPU 2 service jobs | ViT-H+ fold 4 (done), window fold 1 (done) | Done; ViT-H+ fold 2 dropped. |

## Deploy gaps from the integration rehearsal (coordinator, 2026-11-10 10:30 EST)

| # | Gap | Status |
|---|---|---|
| 1 | CV apply function | **Done 11:30 EST**: `source/cv_apply_20261110.py` (`load_params`, `apply`, `ridge`, `apply_fold`). Check reproduces the OOF CV column, max |diff| 2.8e-16. Per-image, label-free parts (graphs from peaks, graph nets, simulation embeddings, token blocks, frozen heads): `source/cv_image_parts_20261110.py`, **done 12:55 EST**, parity on training rows: graphs identical, nets 1.4e-8, embeddings within float16. Wired into the cell-route lane's inference entry. |
| 2 | All-rows fits of the shared `GRAPH` and `BEST` (ridge) | **Done 12:38 EST**: GRAPH nets `graph_models/gfull_{noctx,pre47,pre48}_s<k>_s<k>_e{21..25}.pt` (16 seeds, `source/qC4.sh`); ridge (CV and HEX columns) in `deploy/cv_ridge_params.npz`, strengths CV 30000 / HEX 3000, fit on all good rows against the all-rows GRAPH residual. HEX lane told the paths. |
| 3 | CV deploy stretch, two-seed recipe A scalars | **Done 12:38 EST**: `deploy/cv_deploy_params.json`. Recipe A scalars CD 0.99141, CV 0.98911, HEX 0.99159 (the CD lane stores 0.99139 for CD). CV stretch (0.975, 0.98136, -0.98117). |
| 4 | Which `sg2pre_s47/s48` pair | **Done**: green-track pair (10-30), md5 in FREEZE_CANDIDATE.md. The OOF and all full-data graph nets used fold-detector graphs, so the pair matters only at test time. The cell-route lane agreed. |
| 5 | ids in the OOF file | **Done**: `deploy/cv_final_oof_20261110.npz` (ids, idx, fold, P) and `deploy/cv_inputs_oof.npz` (per-row inputs + per-fold parameters). |

Cannot be done by 11-11 18:00 EST from this lane alone: parity of the full-data members on real images through the end-to-end entry (the cell-route lane owns the entry; my per-image parts are checked on training rows only).

**Lane best update 2026-11-10 12:50 EST: CV 10.1052 (-0.0460)**, adding the order-loss DINOv3 s46 as its own component (five folds, one recipe; out/cv_final_20261110b.npz; FREEZE_CANDIDATE.md updated and reproduced; deploy parameters fit for it).

## 2026-11-11 03:15 EST: freeze candidate decided by the pre-registered rule

- **CV 10.1101 (-0.0411)**, the 6-component stack: DINOv3 seeds 46/47/49, three frozen heads, plain TTA graph quartet, order-loss graph quartet. With the order-loss DINOv3 as the 3-seed mean (46, 47, 49) the stack reads 10.1143 (-0.0369), so the rule kept 10.1101. The 10.1052 one-seed stack is withdrawn.
- deploy/ holds its params (md5 3b437500); the 7-component params are in deploy/_superseded/order7comp_20261111/.
- Night of 11-10 to 11-11, all closed at weight 0 or worse on top of the stack: order-loss ViT-L (folds 0+4), order-loss DINOv2 (folds 0+4), order-loss GRAPH sets (five folds, +0.007), SD-target member (five folds), LDS member (folds 0+4), Vigueras-Guillen segmentation features (five folds, +0.019).
- GPU 2 handed to the backbone lane (coordinator) after seed 49.

## Daily review 2026-11-11 09:00 EST (continue or stop)

| Direction | Used | Decision |
|---|---|---|
| Noise-weighted pairwise order loss | v3 graph quartet ~105 CPU-h; v2 DINOv3 3 seeds ~15 GPU-h; ViT-L ~4 GPU-h; DINOv2 ~2 GPU-h; GRAPH sets ~3 GPU-h | **Stopped.** Kept: the order graph quartet, in the 10.1101 candidate. Dropped: the order DINOv3 (by the pre-registered 3-seed rule: -0.0369 vs -0.0411), ViT-L and DINOv2 (weight 0 on folds 0+4), GRAPH sets (+0.007 on five folds). |
| SD-target member (PI-approved) | ~6 GPU-h | **Stopped.** Weight 0 on folds 0+4; five folds worse (-0.0358 vs -0.0411). |
| LDS member | ~2 GPU-h | **Stopped.** Weight 0 on folds 0+4. |
| Cross-lane members on CV | read-only | Vigueras-Guillen segmentation features +0.019 (five folds), C-Mixup DINOv3-B and attention-pool DINOv3-B weight 0 (folds 0+4). All closed. |
| Deploy | CPU | **Done.** cv_apply + per-image parts with parity checks; all-rows GRAPH and BEST ridge; deploy/ holds the 10.1101 params (md5 3b437500). |

No CV experiment is running. GPU 2 has been handed to the backbone lane. The freeze candidate stays CV 10.1101 unless the coordinator says otherwise.

## 2026-11-11 afternoon (ledger ids in brackets)

- **CV freeze candidate unchanged: D-CV-001**, five folds 10.1101 as computed and 10.1160 strict (best pipeline 10.1512).
- **L5 passes [O-CV-L5-001].** The nested stack code run with no outer fold reproduces deploy/cv_deploy_params.json (md5 3b437500) exactly:
  - weights [0.125, 0.25, 0.375, 0.25, 1.0, 0.25] from both the as-computed and the strict objective;
  - stretch to 1e-16;
  - per fold, the deploy ridge and recipe A scalar recipes reproduce the nested values.
  - Script: source/l5_deploy_check_20261111.py.
- **Synthetic-pretrained fine-tuned DINOv3-B, D-SYN-001: not adopted for CV [O-CV-SYN-002].**
  - Five folds strict, same-run incumbent 10.1160: in place of plain DINOv3 10.1195 (+0.0035); as a 7th component 10.1193 (+0.0034).
  - The folds 0+4 screen had read -0.0110.
  - Script: source/synthft_read_20261111.py.
- **Submission 1 test gap [F-TESTGAP-001].** Inside 1000-image sampling noise (joint tail 0.40); no HEX adjustment is safe. Report: reads/hex_test_gap_20261111.md.
