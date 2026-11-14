# CV lane (GPU 2) notes, 2026-11-08 07:05 EST

**2026-11-11 03:15 EST, FREEZE CANDIDATE: CV 10.1101 (-0.0411)**, the 6-component stack. The coordinator's pre-registered 3-seed rule decided it: with the order-loss DINOv3 as the mean of seeds 46, 47, 49 the stack reads 10.1143 (-0.0369), which does not beat 10.1101. The 10.1052 stack below used seed 46 alone and is withdrawn. `deploy/` holds the 10.1101 params (md5 3b437500). Tonight's other CV tries all read zero or worse on top of the stack: order-loss ViT-L, order-loss DINOv2, order-loss GRAPH sets, the SD-target member, the LDS member, Vigueras-Guillen segmentation features.

**(Withdrawn) 2026-11-10 13:00 EST, lane best (five folds, one recipe): CV 10.1052 (-0.0460)**, per fold -0.101 / +0.055 / -0.073 / -0.063 / -0.047. It is the stack below plus the order-loss DINOv3 s46 as its own component; deploy weights: dino 0, orderdino 0.125. File `out/cv_final_20261110b.npz`. Deploy: `source/cv_apply_20261110.py` with params in `deploy/` and per-image parts in `source/cv_image_parts_20261110.py`. The best pipeline's GRAPH and BEST ridge are now deployable: `graph_models/gfull_*`, `deploy/cv_ridge_params.npz` (CV and HEX columns).

**2026-11-10 07:00 EST, lane best (five folds, one recipe): CV 10.1101 vs 10.1512 (-0.0411)**, per fold -0.103 / +0.055 / -0.059 / -0.063 / -0.035. The order-loss graph quartet (`graph_runs/gord_{s46,s47,s48,seeds3}_tta4_5f.npz`) is added as its own stack component (deploy weight 0.25) next to the plain quartet (1.0). Used instead of the plain quartet it gives -0.0265; averaged with it, -0.0251. File `out/stack5_plain_order_2comp.npz`; FREEZE_CANDIDATE.md updated, reproduction verified, full-data order nets built (CPU pod). The folds 0+4 numbers below came from the screen.

**2026-11-10 22:05 EST 11-09, order loss on the graph network (folds 0+4):** the seeds3_tta4 graph member retrained with 1.0 x noise-weighted pairwise BCE on CV (`source/graph_order_cv_20261110.py`) takes the one-recipe stack from -0.0498 to -0.0620 on folds 0+4 (-0.012; both folds better). Five folds, order versions of the other three graph members and full-data fits are queued on GPU 2 after ViT-H+ (`source/qG2.sh`). Not yet a five-fold number.

**Restated 2026-11-09 (one recipe on all five folds):** lane best **CV 10.114 vs best pipeline 10.151 (-0.037)**, per fold -0.096 / +0.058 / -0.069 / -0.055 / -0.023, file `out/stack5_best_3seeds.npz`. DINOv3 member = mean of seeds 46, 47, 49 on every fold. Deploy weights (fit on all five folds): dino 0.125, spectrum 0.25, cov_sqrt 0.375, autocorr 0.25, TTA graph quartet 1.0. The -0.040 below used seed 47 on folds 0/4 only and is withdrawn.

(Superseded) Lane best, five folds, weights chosen nested (on the other four folds): CV 10.111 vs best pipeline 10.151 (-0.040).
Per fold -0.090 / +0.038 / -0.070 / -0.055 / -0.024. CD and HEX untouched (CV column only).

Stack = best pipeline (`best_pipeline_20261106.py`, BEST) + CV-column log corrections:

| Component | Alone, five folds | Where |
|---|---:|---|
| DINOv3 B/16 member, w (log new - log recipe A TTA); seeds 46+47 on folds 0/4, seed 46 on 1-3 (CD lane) | -0.009 | `lane_gpu3_20261106/runs`, `lane_gpu2_20261106/runs`, `lane_cd_20261108/arms` |
| Frozen token-statistic heads on the recipe A fold encoders (spectrum, cov_sqrt, autocorr), w (lg arm - lg std) | -0.012 | folds 0/4 `lane_gpu2_20261106/reads/own`, folds 1-3 `lane_cv_20261108/reads/own` |
| Graph network on flip-TTA / 3-seed-averaged pre-only peaks (4 members) | -0.006 | `lane_cv_20261108/graph_runs/tta_*_5f.npz` |

Saved: `experiments/lane_cv_20261108/out/stack5_dino_fz_tta.npz` (P = OOF predictions, per-fold weights). Script: `source/cv_stack5_20261108.py`.

Fold 1 gets worse with every image component (+0.04 to +0.08); components show no fold-level offset there.

Running: DINOv3 B/16 with CV-weighted loss (CV x3), seeds 46 and 47, five folds (fold 0: -0.034 vs -0.027 for the recipe-loss member at w .25). Node-selection variants on the averaged peaks.

Dropped after reads (details in `RESULTS.tsv`): spacing-prior node selection (2 versions), 5 windows, box-size windows, window attention (3 versions), fine-tuned-detector graphs, sim-encoder graph members, 5-detector graph ensemble, watershed boundaries, CV-weighted / CV-only frozen heads, DINOv3 frozen heads, ConvNeXt / DINOv2 members on CV, CV from pipeline HEX/CD info, CV-specific A/P1 base weight, CV re-weighting of shipped graph members, CV-weighted recipe A (stopped: 28 h per fold under sharing).

Deployability: every component is a model already in the pipeline family (DINOv3 B/16 member, frozen heads = small MLPs on the recipe A encoder, graph network on detector peaks); all retrain on full data well within 12 h.
