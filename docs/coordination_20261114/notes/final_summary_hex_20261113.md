# HEX lane final summary (2026-11-13, written 21:45 EST)

Lane folder: `/workspace/clear-ec-research/experiments/lane_hex_20261108/` (below: `LH/`). All values are HEX OOF MAPE, five folds (8,952 scored rows) unless folds 0+4 is named.

- "Strict" means the whole procedure was rerun with the scored fold's labels hidden.
- "Chain convention" means the base on the fit rows is the base model fitted on those rows.
- No process, waiter, loop or keepalive of this lane is running (checked 21:43 EST; the loop was stopped 11-12 05:35 EST).

## 1. Outcome

| Item | Strict | vs best pipeline 9.80262 (strict) | Ledger |
|---|---:|---:|---|
| D-HEX-004 freeze candidate: 39-column joint nested ridge (t1 / full) | 9.71977 (nested 9.71882) | -0.0829 | [O-HEX-001, A-HEX-001, A-HEX-002, O-VER-004] |
| same plus the cell-route step (D-HEX-005, deploy lane), end to end | 9.71342 | -0.0892 | [A-DEPLOY-004, O-VER-004] |
| tier t2 = HEX of tier p (no CPSAM multi-target column) | 9.72093 | -0.0817 | [O-HEX-010, A-HEX-006, O-VERIFY-HEX-TIERS] |
| tier p HEX with the cell-route step (t2 base) | 9.71943 | -0.0832 | [O-DEPLOY-014] |
| tier t3 (also no attention pool, cvw3, 16-detector consensus, hex46 net) | 9.73270 (with cell-route 9.73076) | -0.0699 | [O-HEX-010, O-DEPLOY-014] |
| reference: the 10-step chain procedure (former candidate) | 9.74961 (as computed 9.73258) | -0.0530 | [O-HEX-001] |

- Deploy fit of every ridge tier = the nested procedure run with no outer fold. Rerunning it reproduces the stored parameters exactly (L5, max difference 0.0; verifier <= 1.5e-13). [A-HEX-001, A-HEX-006, O-VERIFY-HEX-TIERS]
- The uploaded build is tier p (CD t1 + CV t1 + HEX t2 + cell-route on t2). t3 is the fallback image. [D-PKG-003, A-DEPLOY-012, O-DEPLOY-023]
- Full-data builds made by this lane:
  - cvw3 s46 (2500 s) and s47 (2456 s);
  - CPSAM 512 multi-target (36,079 s);
  - 8 detectors (385-670 s each);
  - graph nets pre47_s10 / pre48_s10 / hex_pre46_s5 (803 / 811 / 401 s).

## 2. Models

"Loaded by" means which packaged tiers read the file: full = t1, then p, then t3.

| Path (pod) | Role | What it is | Size (records) | p | t3 |
|---|---|---|---|---|---|
| `LH/preds/deploy_params_joint_attnp.json` | deployed (full/t1) | 39-column ridge params, sha256 1fbc5384...ad072c2c | small | no | no |
| `LH/preds/deploy_params_joint_t2.json` | deployed (p) | ridge params without mtcnn5 (38 columns), sha256 a6a942f3...f16e2ba6 | small | yes | no |
| `LH/preds/deploy_params_joint_t3.json` | deployed (t3) | ridge params, 34 columns, sha256 744b8ba7...7e81d7f8e | small | no | yes |
| `LH/preds/deploy_params_joint_t3core.json`, `deploy_params_joint.json` (38 col), `deploy_params.json` (chain) | reference / fallbacks | not packaged | small | no | no |
| `LH/deploy/cvw3_f99_s46/runs/fold99/dinov3_b16/epoch_08..12.pt` | deployed (full, p) | CV-weighted DINOv3-B, full data, seed 46 | 3.43 GB with s47 | yes | no |
| `LH/deploy/cvw3_f99_s47/runs/fold99/dinov3_b16/epoch_08..12.pt` | deployed (full, p) | same, seed 47 (seed-mean column) | (above) | yes | no |
| `LH/deploy/mtcps_c512_f99/runs/fold99/cpsam_c512/epoch_08..12.pt` | deployed (full/t1 only) | CPSAM 512 multi-target member (mtcnn5 / cps5), full data | 6.09 GB | no | no |
| `LH/deploy/detectors/{sg2_specialist_s51,sg2_specialist_s52,sg2_sigma3p5,sg2_sigma5}/detectors/*.pt` | deployed (full, p) | 4 of the 16 detectors of the ridge_fall16 consensus | 1.37 GB with heads | yes | no |
| `LH/deploy/frozen_heads/{sg2f_s46,sg2f_s47,sg2f_s48,sg2pref_s46}/fold0/head.pt` | deployed (full, p) | 4 heads on the recipe A full-data encoder (fall16 consensus) | (above) | yes | no |
| `LH/deploy/graph_models/full_d64_l5_pre47_s10_*`, `full_d64_l5_pre48_s10_*` | deployed (all) | graph nets of G_s10x3, together with the CV lane's gfull_noctx | small | yes | yes |
| `LH/deploy/graph_models/full_hex_pre46_s5_s*_e*.pt` | deployed (full, p) | HEX-only graph net, 5 seeds x 5 epochs | 0.01 GB, 25 files | yes | no |
| `LH/deploy/vigueras/tvst_{edge,roi}.pt` | closed, reusable | PyTorch port of the Vigueras-Guillen TVST CNN-Edge/ROI (GPL-3.0) | ~57 MB | no | no |
| `LH/runs/mtcps_nossl_c512_f{0..4}/` | candidate evidence | fold models of the CPSAM member; their predictions are the OOF mtcnn5 column | not recorded | no | no |
| `LH/runs/cvw3_f{0..4}_s47/` | candidate evidence | seed-47 fold runs; their predictions are half of the OOF cvw3 column | not recorded | no | no |
| `LH/runs/mtcps_nossl_c512_f{0,4}_s47/` (f1, f2 partial, stopped) | closed evidence | mtcnn5 seed-audit runs | not recorded | no | no |
| `LH/runs/hexcps_c512_f4/` | closed | HEX-weighted CPSAM loss | not recorded | no | no |

Other lanes' weights the HEX tiers read (owner lanes keep them):
- **All tiers:** green track hex6 (9 detectors, `green_track_20261028/deploy/`); the pre-only detectors `green_track_20261028/deploy/detectors/sg2pre_s{46,47,48}.pt` (prior, ridge k3, graphs); the CV lane's BEST/GRAPH (`lane_cv_20261108/deploy/cv_ridge_params.npz`, gfull graph models); `cpsam_models/cpsam_v2` (zero-shot segmentation).
- **full and p:** the green track's `sg2_s46..50` (fall16 consensus); the backbone lane's attention pool `lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46/raw/runs/fold99/dinov3_b16/{epoch_08..11,final}.pt` (1.77 GB).

## 3. Data

| Path | What | Regenerable? | Also read by |
|---|---|---|---|
| `LH/preds/chain_joint_attnp.npz` / `_strict.npz` | t1 OOF HEX, nested 9.71882 / strict 9.71977 | `source/joint_deploy.py build` (CPU) / assembled from `runs/strict/` | deploy lane, verifier |
| `LH/preds/chain_joint_t2.npz`, `chain_joint_t3.npz`, `chain_joint_t3core.npz` | tier OOFs (nested) | `JOINT_TAG=.. JOINT_DROP=.. source/joint_deploy.py build` | deploy lane (cell-route refits) |
| `LH/preds/chain_joint.npz`, `chain_seedaudit10.npz`, `chain_procedure_strict.npz` | fallbacks / references | `joint_deploy.py`, `chain.py`, `strict_nest.py` | verifier |
| `LH/preds/seg_all.npz` | zero-shot Cellpose-SAM features per frame (17) | `source/seg_run.py` + `seg_merge.py` (GPU, hours) | cell-route lane (`deploy_hexkonan`) |
| `LH/preds/hex_pre46_s5.npz`, `hex_pre47_s5.npz`, `hex_pre4647_s5.npz` | graph-net OOF (5 folds) | `source/gnn_hex.py` (GPU, ~30 min each) | none |
| `LH/preds/hex_inputs_oof.npz` | OOF input dict of `hex_apply` | `hex_apply.py oof` | deploy lane |
| `LH/feats/prior_pre3.npz`, `ridge_k3.npz`, `ridge_fall16.npz` (+ fpk_*, g_*) | OOF feature caches | `feat_prior.py`, `graph_prep_lane.py`, `feat_ridge.py`, `fuse_peaks.py` (CPU, from green-track fold peak files) | none |
| `LH/preds/seg_vig_all.npz`, `LH/runs/vig_maps/` | Vigueras features / edge maps (closed) | `vig_infer.py` + `vig_post.py` | CV lane read it once |
| `LH/runs/strict*`, `spm_*`, `swapA*`, `RESULTS.tsv`, `REPORT.md`, `FREEZE_CANDIDATE.md`, `logs/` | strict fold runs, reads, records | scripts in `LH/source/` | coordinator, verifier |

Dependencies, not made here:
- best-pipeline parts: `recipeA_analysis_20261028/hybrid_oof.npz`, `submission2_search_20261103/green_features.npz`, `meta_ceiling_20261103/sweep/{sim_pred,graph_sweep,cell_graphs*.npz}`;
- `eval_harness_20261031/image_stats.npz`;
- `datasets/clear_ec_fullframe_20260904/img_full_raw.dat`;
- green-track fold peak files;
- the CV lane's cvw3 s46 arms and `deploy/cv_deploy_insample_members.npz`, which supplies the in-sample BEST of the deploy fit;
- the backbone lane's attention-pool fold runs.

## 4. Ideas

(a) Adopted (strict, five folds):
- `idea-joint-nested-ridge-hex-stack`: all HEX corrections in one nested ridge instead of a sequential chain. 9.72662 (38 col) vs chain procedure 9.74961. [O-HEX-001]
- `idea-attention-pool-column-hex`: attention-pool DINOv3-B column in the ridge, 9.71977 vs 9.72662. Read as a chain member at share 0.2 it was +0.0055 on folds 0+4. [O-HEX-001, O-HEX-010]
- `idea-zero-shot-cpsam-segmentation-features`: the largest block. The ridge without it reads 9.75186 nested; leave-one-out in the chain +0.0252. [REPORT.md]
- `idea-cpsam-multitarget-member`: costs +0.0012 strict when removed. The second seed keeps the gain (-0.0011 / -0.0015 / -0.0016 for s46 / s47 / mean, nested). [O-HEX-004, O-HEX-010]
- `idea-cv-weighted-dinov3-seed-mean`: costs +0.0031 when removed. Under the seed-audit rule s46 alone was +0.0027 and the s47 + s46 mean -0.0009. [O-HEX-001]
- `idea-strict-nesting-and-deploy-equals-nested-procedure`: reporting and deploy standard (L4, L5). [A-HEX-001]

(b) Closed:
- `idea-vigueras-guillen-edge-network`: +0.0003 (chain, five folds); CV lane +0.0196. [RESULTS.tsv 11-11]
- `idea-synthetic-zero-shot-columns`: +0.00109. [O-HEX-002]
- `idea-synthetic-pretrained-dinov3b-member`: +0.00101 (two folds). [O-HEX-003]
- `idea-synthetic-pretrained-recipeA`: +0.0017 (five folds; -0.0034 on folds 0+4 only). [O-HEX-008, O-HEX-009]
- `idea-cmixup-dinov3b-member`: +0.0153 on folds 0+4 in the chain; +0.00108 in the ridge. [RESULTS.tsv]
- `idea-forward-selection-from-scratch`: 9.74827 vs chain 9.73258 / ridge 9.72621. [REPORT.md]
- `idea-gnn-detector-seed-mean`: +0.00003, dropped in the chain. It stays in t1/p as a single detector seed (+0.0009 when removed in the ladder). [O-HEX-010]
- `idea-16-detector-consensus-ridge`: removal costs -0.0002 in the t3 ladder, i.e. free. [O-HEX-010]
- Earlier closures, recorded in RESULTS.tsv and REPORT.md: segmentation variants (segv, segm2, segr, segb, segg), the HEX-weighted CPSAM loss, CD-lane ViT-H+/16 (+0.029 on folds 0+4), shipped DINOv3 (+0.0034), backbone win2x (+0.0002).
- Leave-one-out cascade: `ridgeev` read -0.00046, not acted on (L7). [REPORT.md]

(c) Open or partly tested:
- `idea-attention-pool-seed-audit`: the adopted column rests on seed 46 only. L3 was not applied to it (no second seed).
- `idea-deploy-base-convention`: the ridge reads 9.72662 under the chain convention and 9.74136 under an OOF-input base. The deploy fit uses the CV lane's all-rows BEST in-sample (G + RID, r 0.990 with the OOF BEST) as the chain-convention analogue. Its effect on test is unmeasured.
- `idea-hex-test-gap`: submission 1's test-minus-OOF gap is HEX +0.395.
  - It lies inside 1000-image sampling noise (SD of the gap 0.257; HEX alone at the 6.4 % tail), and no HEX offset is safe [F-TESTGAP-001].
  - Leave-one-source-out costs HEX +3.5 / +0.8 MAPE points, mostly a level offset [F-SHIFT-001].
  - This lane did not test any HEX test-time adjustment.
- `idea-t4-precision-hex`: on the T4 path the HEX deviation is 0.44-0.45 % median (max 3.9 %), estimated HEX impact +0.018. fp32 does not reduce it. [O-DEPLOY-020/021, O-COORD-F32-200]

## 5. Folder map

| Folder | Purpose | Verdict | Reason |
|---|---|---|---|
| `LH/source/` | all HEX scripts (chain, joint_fit, joint_deploy, hex_apply, strict_*, feature builders) | KEEP | reproduces D-HEX-004 and the tiers |
| `LH/deploy/` (cvw3_f99_*, mtcps_c512_f99, detectors, frozen_heads, graph_models) | full-data weights | KEEP-MODEL | packaged in full/p/t3 |
| `LH/deploy/vigueras/` | ported Vigueras weights | ARCHIVE | closed, small |
| `LH/preds/` | params, OOF files, seg_all, graph-net OOF | KEEP | packages and reproduction |
| `LH/feats/` | OOF feature caches | KEEP | needed to reproduce the OOF without GPU |
| `LH/runs/mtcps_nossl_c512_f{0..4}/`, `runs/cvw3_f{0..4}_s47/` | fold runs whose predictions feed the OOF columns | KEEP predictions; checkpoints DELETE-AFTER-SUBMISSION | packages use only the f99 builds |
| `LH/runs/*_s47` (mtcps), `runs/hexcps_c512_f4/` | seed-audit and closed fold runs | DELETE-AFTER-SUBMISSION (keep predictions_*.npz) | evidence only |
| `LH/runs/strict*`, `spm_*`, `swapA*`, `vig_sanity` | strict reads | ARCHIVE | small evidence |
| `LH/runs/vig_maps/` | Vigueras edge/ROI maps for 8,954 frames | DELETE-NOW | closed, regenerable, large |
| `LH/seg_loose/` | Cellpose-SAM mask chunks behind seg_all | DELETE-AFTER-SUBMISSION | seg_all.npz holds the features |
| `LH/seg_rot/` | rotated segmentation run (closed) | DELETE-NOW | closed, masks only |
| `LH/tmp_pseudo_test*`, `LH/tmp_gnn_check_*.npz` | parity test outputs | DELETE-NOW | scratch |
| `LH/pylib/` | h5py 3.11.0 used once to convert the Vigueras HDF5 | DELETE-NOW | not needed by any package |
| `LH/archive/`, `LH/logs/`, `REPORT.md`, `RESULTS.tsv`, `FREEZE_CANDIDATE.md` | records | ARCHIVE | evidence trail |
| `green_track_20261028/deploy/` | hex6 bundle and detectors (sg2pre, sg2_s46..50, s8*) | KEEP-MODEL | packages load hex6, pre3, sg2 |
| `green_track_20261028/{external_srep,sg2_specialist,frozen_heads}/` | fold detector runs and peak files | KEEP until the caches are archived | `LH/feats` and `det_peaks_image.py --verify` read them |
| `green_track_20261028/source/` | `green_point_features_20261027.py`, `green_sg2_specialist_20261028.py`, `green_selftrain_20261028.py` | KEEP | imported by `det_peaks_image.py` and `hex_test_features.py` (packages) |
| `green_point_features_20261027/` | point-feature extraction runs (extract, select, specialist) | ARCHIVE | not read by HEX code; the module of the same name lives in green_track source |
| `green_point_probe_20261023/` | an early probe run (head_terminal200.pt) | ARCHIVE | not read by HEX code |
| `/workspace/scratch_hex6_bundle/` (+ `_sums.txt`) | copy of a hex6 deploy bundle (ACTIVE.json, detectors) | check before moving | it is not known here whether the deploy manifests point at it or at green_track deploy |
| `meta_ceiling_20261103/`, `recipeA_analysis_20261028/`, `submission2_search_20261103/`, `eval_harness_20261031/` | best-pipeline parts and inputs | KEEP | `hexlib.py`, `chain.py` and `hex_apply.oof_inputs` read them |
| `fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt` | encoder of the 4 frozen heads | KEEP-MODEL | `det_peaks_image.py` FULL constant |
| `lane_cv_20261108/arms/dinov3_cvw3_f*_s46/`, `deploy/` | cvw3 s46 OOF; BEST/GRAPH and in-sample members | KEEP | `chain.py`, `hex_apply.py`, `joint_deploy.py` |
| `lane_backbone_20261108/runs/attnpool_f*_dinov3b16_s46/` | attention-pool OOF and f99 weights | KEEP / KEEP-MODEL | ridge column (full, p) |
| `datasets/external_20261106/vigueras_*` | downloaded Vigueras code and weights (GPL-3.0) | ARCHIVE | closed |

By-name references that break if folders move:
- `LH/source/*.py`: `LANE` / `XR` constants (`/workspace/clear-ec-research/experiments/lane_hex_20261108`, `.../experiments/`). `chain.py` hard-codes the lane_cv, lane_backbone, meta_ceiling and green_track paths. `hex_apply.py` reads `LH/preds/*`, `LH/feats/*`, `hex_pre46_s5.npz`. `joint_deploy.py` reads `lane_cv_20261108/deploy/cv_deploy_insample_members.npz` and the attention-pool fold runs.
- `det_peaks_image.py`: its FULLDATA dict hard-codes `green_track_20261028/deploy/detectors/*.pt`, `LH/deploy/detectors/*`, `LH/deploy/frozen_heads/*` and the fulldata_recipeA encoder.
- `gnn_hex_infer.py`: `LH/deploy/graph_models`.
- `family_infer.py`: imports `LH/source/family_cvw3_copy.py`.
- Deploy lane:
  - `params/deploy_*.json`, `manifests/deploy_*.json` and `params/cellroute_deploy_*.json` name the `LH/preds/deploy_params_joint*.json` files and the `LH/deploy/` weights.
  - The staged `package_20261112/<tier>/{context,model}` trees are hard links, so moving the originals does not break them; re-staging does.
- Cell-route lane: `deploy_hexkonan` reads `LH/preds/seg_all.npz` and imports `seg_run.py`.

## 6. Loose ends

- Unverified on test: the deploy-convention choice (section 4c), and the single seed behind the attention-pool column.
- Tier p ran 287-288 s on the first Colab T4 [O-DEPLOY-T4-COLAB] and 310.7-314.2 s on a second one [O-COORD-COLAB-T4-B], which is over the 300 s limit. t3 ran 173-174 s on the first. These are deploy-lane measurements, reported here because HEX t2/t3 are part of p/t3.
- Ledger corrections already filed:
  - [O-HEX-007b]: time stamp of A-HEX-005.
  - The 11-11 10:00 EST "joint ridge not robust" message was withdrawn at 10:25 EST: it compared a deploy-convention number with a chain-convention one. That became lesson L6.
- `RESULTS.tsv` holds rows marked "kept (forced)" from 11-11 ~09:28Z. These are leave-one-out audit reads, not chain decisions; an annotation row follows them.
- The strict values keep the best pipeline's upstream constants as given (h6 blend 0.28, f6 map -0.2084 + 0.5989 f6). They were set on all rows and are shared by every number here.
- Nothing was deleted, moved or renamed by this lane in this closing step.
