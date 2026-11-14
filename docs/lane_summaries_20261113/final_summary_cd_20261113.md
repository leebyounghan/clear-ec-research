# CD lane: final summary (2026-11-13, 21:50 EST)

CD lane, GPU 0. Lane folder: `/workspace/clear-ec-research/experiments/lane_cd_20261108/`; CPU pod `clear-ec-cpu-1109`.

- All paths below are relative to `/workspace/clear-ec-research/experiments/` unless they start with `/`.
- MAPE is in %. "Strict" = every fitted constant (P1 and recipe A scalars, weights, mask slot, stretch) is fit on the four folds other than the scored one.
- The best pipeline is the reference: 6.2639 as computed, 6.2610 strict.

## 1. Outcome

**Frozen CD candidate [D-CD-001].** The stack is:
- log blend of P1 × SCALE 0.25, recipe A s46+s47 snapshot 0.45, DINOv3-B s46+s47 seed-mean TTA (epochs 16-20) 0.10, and ViT-L/16 s46+s47 ("tt" form) 0.20;
- then the seg_loose mask density `cd_mm2` (rows with n_int ≥ 50, weight 0.1);
- then a MAPE-optimal log-affine stretch.

Scores:
- Five folds: 6.1741 as computed, **6.1743 strict**, against the best pipeline's 6.2610 strict: **−0.087**.
- Every fold is lower than the best pipeline: −0.101 / −0.066 / −0.052 / −0.117 / −0.112 [O-CD-SYN5-READ incumbent line].
- Files: `reads/freeze_cd_{oof,params}_20261111.*`; `FREEZE_CANDIDATE.md`.

**Deploy tiers [D-PKG-002, O-CD-TIERS, A-CD-TIER-T1/T2/T3].** Incumbent recomputed in the same run:

| CD tier | What changes | Strict CD | vs incumbent |
|---|---|---:|---|
| t1 | ViT-L at final.pt TTA only; recipe A at e20 TTA only | 6.1737 | −0.0006 (CI −0.007 to +0.006) |
| t2 | t1 without ViT-L | 6.1875 | +0.013 |
| t3 | P1 + recipe A e20 + mask only | 6.2138 | +0.040 |

- CD t1 is inside tier p and tier t1 [D-PKG-003, D-PKG-005]; CD t3 is inside tier t3.

**Full-data builds made on GPU 0:**
- DINOv3-B s46b (1.9 h), DINOv3-B s47 (1.1 h), ViT-L s46 (4.0 h).
- ViT-L s47 was built from this lane's queue line on GPU 3 by the backbone lane.
- All four were verified: sha256, strict load, held rows reproduced (`reads/verify_f99_{vitl,dinov3}_20261111.json`).

**Verified for other lanes:**
- 11-11 claims table: every lane's value reproduces within 0.0005 [O-VER-001, O-VER-002]. L5 for CD [F-VER-001]. HEX strict 9.71977 assembled independently [O-VER-004, O-VER-005]. Cell-route L5 and pick [F-VER-002].
- 11-12 CV tiers: incumbent 10.11596, t1 10.1115, t3 10.11379, t3_alt 10.12378, all exact; L5 passes [O-VERIFY-CV-TIERS].
- 11-12 HEX strict ladder, rerun from scratch: t1 9.71977, t2 9.72093, t3 9.73270, t3core 9.73200; L5 ≤ 1.5e-13 [O-VERIFY-HEX-TIERS].
- Cell-route end-to-end on HEX t2 / t3: 9.71943 / 9.73076.
- Tier strict means: p 8.66820, full 8.66789, t1 8.6662.

**Not verified:**
- the cell-route tier deploy params (`lane_cellroute_20261109/params/cellroute_deploy_t*.json`) against a refit with no outer fold;
- the deploy lane's packaged and T4 parity and timing (their own checks);
- the p / t3 container tests.

## 2. Models

Sizes are fp32 from my records; fp16 copies sit at `lane_cellroute_20261109/deploy_fp16/<same path>`. Marks: **P** = loaded by the tier-p package, **T3** = loaded by the tier-t3 package, F = full candidate only.

| Path (pod) | Role | What it is | Size |
|---|---|---|---|
| `lane_cd_20261108/arms/dinov3_b16_f99_s46b/runs/fold99/dinov3_b16/epoch_16..20.pt` | **P** (CD D and CV t1 term); **T3** (epoch_20.pt only, CV t3 term); F | DINOv3-B/16 s46 full data (train 8952) | 5 × 343 MB |
| `lane_cd_20261108/arms/dinov3_b16_f99_s47/runs/fold99/dinov3_b16/epoch_16..20.pt` | **P** (CD D); F | DINOv3-B/16 s47 full data, SSL reused from s46 | 5 × 343 MB |
| `lane_cd_20261108/arms/vitl16_f99_s46/raw/runs/fold99/dinov3_l16t/final.pt` | **P** (CD V, t1 form); F | ViT-L/16 DINOv3 s46 full data, epoch 12, sha256 fbb0eca8… | 1.21 GB |
| `lane_cd_20261108/arms/vitl16_f99_s47/raw/runs/fold99/dinov3_l16t/final.pt` | **P**; F | ViT-L/16 s47 full data, sha256 d43c3a73… | 1.21 GB |
| `lane_cd_20261108/arms/vitl16_f99_s{46,47}/raw/runs/fold99/dinov3_l16t/epoch_08..11.pt` | F only (not used by t1 / p) | ViT-L snapshot epochs | 8 × 1.21 GB |
| `lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt` | training provenance (SSL init of s46b and s47; not read at inference) | DINOv3-B SSL5 on 8952 rows | n/r |
| `lane_cd_20261108/arms/dinov3_b16_f{1,2,3}_s{46,47}`, `vitl16_f{1,2,3}_s46`, `vitl16_f{0..4}_s47` | candidate (fold runs; their predictions are the OOF members) | per-fold members; checkpoints are not deployed | ViT-L 5 × 1.21 GB per arm |
| `lane_cd_20261108/arms/convnext_b32_f{1,2,3}_s46`, `dinov3_dist_f{0..4}_s46`, `vith16_f{0,4}_s46`, `cdonly_{ra,elr3e5_ra}_f{0,4}_s46` | closed, reusable (predictions) | ConvNeXt-B, distribution head, ViT-H+, CD-only recipe A | n/r |
| (not this lane) `fulldata_recipeA_20261028/regression_std_s{46,47}/runs/fold99/cpdino_vitb/epoch_20.pt` | **P**, **T3** (CD A e20; CV/HEX A) | recipe A full data | 343 MB each |
| (not this lane) `fulldata_recipeA_20261028/regression_std_s{46,47}/.../epoch_08..19.pt` | F only | recipe A snapshot epochs | 24 × 343 MB |
| (not this lane) `deploy_review_20260905/staged/model/{seed42,seed77,seed2026,metric35,predbag2_35}.pt` | **P**, **T3**, F (P1) | crop CPSAM members; sha256 equal to submission 1's | 5 × 1.22 GB |
| (not this lane) `/workspace/clear-ec-research/cpsam_models/cpsam_v2` | **P**, **T3**, F (P1 backbone and seg mask) | Cellpose-SAM | 1.23 GB |

## 3. Data

| Path | What it is | Regenerable from | Also read by |
|---|---|---|---|
| `lane_cd_20261108/reads/freeze_cd_oof_20261111.npz`, `freeze_cd_params_20261111.json` | D-CD-001 OOF inputs, per-fold and deploy params | `source/cd_freeze_20261111.py` (CPU, seconds) | deploy lane `inference/params_build.py` (path hard-coded) |
| `lane_cd_20261108/reads/freeze_cd_params_t{1,2,3}_20261112.json`, `cd_tier_{inc,t1,t2,t3,t1d,t2d}_20261112.{npz,json}`, `cd_tiers_summary_20261112.json` | CD tier params, strict OOF, member inputs | `source/cd_tiers_20261112.py` + `cd_tiers_assemble_20261112.py` | deploy lane tier builds (params/deploy_<tier>.json) |
| `lane_cd_20261108/reads/strict_nesting_20261111.npz` | strict OOF of v3 and of the best pipeline | `source/cd_strict_nesting_20261111.py` | anatomy, verifier |
| `lane_cd_20261108/tta/*.npz` (120 files) | four-flip TTA predictions per fold (DINOv3 s46/s47 e16-20 = the D member OOF; ConvNeXt, dist head) | `source/family_tta_20261108.py`, **needs the fold checkpoints** | `cd_read_20261108.member()` |
| `lane_cd_20261108/arms/*/raw|runs/fold*/*/predictions_*.npz` | per-epoch and final-TTA OOF predictions of every member | only by retraining | CD reads, CV lane `cv_tiers_20261112.py` (DINOv3 fold runs) |
| `lane_cd_20261108/reads/recipeA_swapped_syn46_20261111.npz` (sha256 c5c1e7c1…) | recipe A with synthetic s46 swapped in | `source/build_swapped_recipeA_20261111.py` | CV and HEX lanes (closed, D-SYN-008) |
| `lane_cd_20261108/reads/verify_*.json`, `hex_ladder_verify_20261112/` | verifier outputs | `source/verify_*_2026111{1,2}.py` | coordinator |
| `lane_cd_20261108/reads/cd_error_anatomy_20261111.{json,md}`, `cd_levers_20261111.json`, `cd_sub1_calib_20261113.json`, `cd_phase1_calib_20261113.json`, `cdtrain_*` | evidence of closed ideas | scripts of the same name in `source/` | — |
| `lane_cd_20261108/feats/seg_loose_area_stats.npz` | mask-area statistics cache (11-09 mask variants) | from `lane_hex_20261108/seg_loose` | — |
| (dependency) `recipeA_analysis_20261028/hybrid_oof.{npz,json}` | P1 OOF (`historical_five_calibrated`), recipe A OOF and LOFO scalars, submission 1 OOF (`B_hybrid_w0p5`) | not by me | every CD script (`cd_read_20261108.py`) |
| (dependency) `meta_ceiling_20261103/source/{best_pipeline_20261106,stretch_nested_20261103,ceiling_meta_cpu_20261103}.py` and its cached arrays | labels, folds, `good` mask, REF, `C.RECIPE` and `HIST` path tables | — | all lanes |
| (dependency) `oof_recipeA_20261027`, `phase2_pooling_20261020`, `phase2_pooling_replication_20261026` | recipe A per-fold runs (`C.RECIPE`) | retraining | all lanes |
| (dependency) `followup_20260902`, `submission_20260904`, `methods_20260904` `runs/*/best_oof.npz` | P1 five members' OOF (`HIST`) | retraining | P1 OOF; tonight's Phase I read |
| (dependency) `lane_hex_20261108/seg_loose/c*.npz` | `cd_mm2`, `n_int` mask features | HEX lane seg run | CD stack (`cd_stack_20261109.py`) |
| (dependency) `lane_gpu3_20261106/runs/dinov3_b16_f{0,4}_s46`, `lane_gpu2_20261106/runs/dinov3_b16_f{0,4}_s47`, `lane_backbone_20261108/runs/vitl16_avg_f{0,4}_s46` | folds 0 and 4 of the D and V members | retraining | CD and CV lanes |
| (dependency) `error_case_analysis_20261106/{peak_traits.npz,tables/forensics_rows.npz}`, `lane_gpu0_20261106/source/calib2_20261106.py` | detector traits, image-quality estimators | — | anatomy only |
| (dependency) `raw_ssl_next_20260913/ssl_runs/fold{2,3,4}/standard/oof.npz` | Phase I sixth member, development OOF | — | O-CD-PHASE1-CALIB |

## 4. Ideas

### (a) Adopted

| Idea | Evidence |
|---|---|
| Geometric P1/recipe A reweighting | −0.035 vs hybrid B, cross-fit (11-08 R2) |
| Other-family member: DINOv3-B/16 | −0.047 five folds (11-08) |
| Four-flip TTA over epochs 16-20 for family members | −0.008 over single view (11-08) |
| Recipe A snapshot (½ e20 TTA + ½ e8-19 single) | −0.004 as computed (11-08); **zero under strict** (t1 = full, O-CD-TIERS) |
| ViT-L/16 DINOv3 member | stack 6.1775, −0.086 vs best pipeline (11-09) |
| ViT-L seed mean (s46+s47) | −0.0049 five folds (11-09) |
| Mask-density CD slot (`cd_mm2`, n_int ≥ 50, w 0.1) | −0.017, all five folds (11-09) |
| DINOv3 two-seed mean (seed-audit rule, no seed picked) | +0.0016; adopted by rule (O-CD-001) |
| Strict-nesting check | v3 +0.0002, best pipeline −0.0028 (11-11) |
| Deploy-size tiers (t1 simplification) | t1 −0.0006 strict (O-CD-TIERS) |

### (b) Closed, with the number that closed it

| Idea | Number |
|---|---|
| Arithmetic P1/recipe A reweight | −0.006, not significant |
| Per-source stretch / per-source blend weights | −0.009 on B, +0.022 after the family blend / +0.012; per-source choices break the charter |
| Cell-graph / simulation-embedding CD column | 0 to +0.010 |
| Detector-density component | weight 0; source-B-only −0.001, dropped by charter |
| ConvNeXt-B/32 member | weight 0.05, ~0.000 beyond DINOv3 and ViT-L |
| DINOv3 stride 8 | member 6.88 vs 6.58 |
| Recipe A third seed s48 | −0.003 on two folds; 26 GPU-h |
| CV-weighted DINOv3 (cvw3) as CD member | −0.048 vs −0.055 without it |
| Member-disagreement MAPE shift | +0.012 to +0.025 [F-CDANAT-001] |
| Ridge on CD residual from mask features | +0.024 |
| Mask-density variants, rotated segmentation, Voronoi CD | same as the plain mean (−0.009 to −0.018; +0.000) |
| ViT-L third seed | s47 adds 0.000 on folds 0, 1, 2, 4 |
| Distribution head (HL-Gauss) DINOv3 | +0.008 five folds |
| DINOv3 ViT-H+/16 | −0.0034 on two folds averaged into ViT-L; closed for cost |
| Technician-window crop recipe A | +0.0026 five folds (4 folds −0.0041) |
| C-Mixup DINOv3 | +0.0099 added |
| Attention-pool DINOv3 | +0.0047 added / +0.0040 replacing |
| From-scratch combination search M1/M2 | 6.1733 / 6.1767 vs 6.1741 |
| Synthetic-only zero-shot DINOv3; synthft3 DINOv3 | weight 0 |
| Synthetic-pretrained ViT-L s46 | −0.0031 five folds strict, under seed noise [O-CD-SYN4-001, D-SYN-006] |
| Synthetic-pretrained recipe A s46 | −0.0017 (CI −0.012 to +0.008) five folds [O-CD-SYN5-READ, D-SYN-008] |
| Label-free Fourier-ring CD | weight 0 [O-CD-FFT-001] |
| Native-tile CD-only DINOv3 | +0.0001 added [O-CD-TILE-001] |
| CD-only loss recipe A (encoder LR 1e-5 / 3e-5) | −0.0030 / +0.0181 folds 0+4; better alone, not in the stack [O-CDTRAIN-1E5-F04, O-CDTRAIN-3E5-F04, F-CDTRAIN-3E5-WHY] |
| Cheap post-hoc levers (global / per-level multiplier, clipping, mask at full weight) | +0.0046 / +0.0050 / +0.038 / +1.35 [F-CDANAT-001] |
| Recalibration against submission 1 / blending submission 1 | recalibrations +0.004 to +0.050; blend +0.0024 (+0.0011 with stretch); sub-1 is spanned (0.5 A + 0.5 P1) [O-CD-SUB1-CALIB] |
| Recalibration against Phase I best / blending it | folds 2-4: Phase I 6.591 vs candidate 6.239; blend +0.0025; sixth member −0.0005, not significant [O-CD-PHASE1-CALIB] |

### (c) Open or partly tested

| Idea | Status |
|---|---|
| Label scatter within P1 donors | within-donor residual 0.083 vs 0.043 (P2), a proxy only: the ids do not say which eye [O-CDANAT-002] |
| Impossible-label frames | 6 rows (2020-0644..0651) are worth 0.187 MAPE; unpredictable; no action taken [O-CDANAT-001] |
| Synthetic recipe A seed mean on folds 1-4 | fold-0 seed mean −0.022 (CI −0.057 to +0.016) [A-BB-SYN7-RESULT]; not run |
| SWA average instead of epoch snapshots | not run; moot after t1 |

## 5. Folder map

Labels follow the coordinator's scheme: KEEP, KEEP-MODEL, ARCHIVE, DELETE-AFTER-SUBMISSION, DELETE-NOW.

| Folder | Purpose | Label and reason |
|---|---|---|
| `lane_cd_20261108/arms/dinov3_b16_f99_s46b`, `dinov3_b16_f99_s47`, `vitl16_f99_s46`, `vitl16_f99_s47` | deployed full-data weights | **KEEP-MODEL** (p, t3, full manifests) |
| `lane_cd_20261108/arms/dinov3_b16_f99_s46` | SSL init of the above | ARCHIVE (provenance; ssl/last.pt is needed to rebuild s47) |
| `lane_cd_20261108/arms/` fold arms of adopted members | OOF members | KEEP the `predictions_*.npz` and `complete.json`; checkpoints DELETE-AFTER-SUBMISSION (they are only needed to regenerate `tta/`) |
| `lane_cd_20261108/arms/` closed arms (convnext, dist, vith16, cdonly) | closed members | predictions ARCHIVE; checkpoints DELETE-AFTER-SUBMISSION |
| `lane_cd_20261108/arms/_archive_cvlane/` | CV lane's stopped ViT-H+ f2 run (put here by the CV lane) | DELETE-NOW (partial; CV lane's call) |
| `lane_cd_20261108/tta/` | D member OOF (TTA) | **KEEP** (stack inputs; regeneration needs fold checkpoints) |
| `lane_cd_20261108/reads/` | freeze, tier, verifier and evidence files | **KEEP** (freeze_cd_* and tier files are read by the deploy lane); rest ARCHIVE |
| `lane_cd_20261108/source/` | 83 scripts | **KEEP** (FREEZE_CANDIDATE.md and the deploy lane cite them; CV and HEX lanes ran some of them) |
| `lane_cd_20261108/logs/`, `RESULTS.tsv`, `REPORT.md`, `FREEZE_CANDIDATE.md` | records | ARCHIVE (small) |
| `lane_cd_20261108/archive/` | failed, stopped and smoke runs; superseded freeze_v1 | `freeze_v1_vitl_s46` ARCHIVE; `failed_*`, `stopped_*`, `_smoke*`, `_check_vitl_*`, the vith16 partial DELETE-NOW (partial runs) |
| `lane_cd_20261108/queue*.txt`, `STOP*`, `feats/` | runner control files, mask cache | DELETE-NOW (control files); feats ARCHIVE |
| `meta_ceiling_20261103` | best pipeline, labels, folds, stretch, path tables | **KEEP** (imported by every lane's scripts) |
| `recipeA_analysis_20261028` | hybrid_oof (P1, recipe A OOF, LOFO scalars, sub-1 OOF) | **KEEP** |
| `fulldata_recipeA_20261028` | recipe A full-data checkpoints | **KEEP-MODEL** for epoch_20 (p, t3) and epoch_08..19 (full); epoch_01..07 and last.pt are in no manifest: DELETE-AFTER-SUBMISSION |
| `oof_recipeA_20261027`, `phase2_pooling_20261020`, `phase2_pooling_replication_20261026` | recipe A fold runs (`C.RECIPE`) | KEEP predictions; checkpoints DELETE-AFTER-SUBMISSION |
| `deploy_review_20260905` | `staged/model`: the five P1 members, candidate5.json, the Phase I staged inference | **KEEP-MODEL** (`staged/model` is in every manifest); the rest ARCHIVE |
| `submission1_test_20261030` | P1 / recipe A inference code (`phase1_app`, `hybrid_inference_20261030.py`) used for submission 1 | KEEP (FREEZE_CANDIDATE.md points the deploy entry at it; check the deploy lane's context before moving) |
| `followup_20260902`, `submission_20260904`, `methods_20260904` | Phase I runs; `best_oof.npz` of the P1 members (`HIST`) | KEEP the `best_oof.npz`; checkpoints ARCHIVE or DELETE-AFTER-SUBMISSION (deploy uses staged/model) |
| `raw_ssl_next_20260913` | Phase I sixth member (SSL + raw) development runs, raw-only package | ARCHIVE (Phase I legacy; only tonight's read uses its oof.npz) |
| `region_candidate_20260907`, `deploy_methods_20260905` | Phase I candidates (candidate6, older predbag2_35) | ARCHIVE or DELETE-AFTER-SUBMISSION (do not use the older predbag2_35) |
| `lane_gpu3_20261106`, `lane_gpu2_20261106` | DINOv3 folds 0/4 s46 / s47 runs | KEEP predictions (D OOF, CV dino term); checkpoints DELETE-AFTER-SUBMISSION |
| `lane_hex_20261108/seg_loose` | mask features | **KEEP** |
| `error_case_analysis_20261106`, `lane_gpu0_20261106` | traits, image-quality code (anatomy) | ARCHIVE |
| `integration_rehearsal_20261110` | REHEARSAL.md with the verifier tables | ARCHIVE |
| `eval_harness_20261031`; `/workspace/{artifacts,checkpoints,logs,scratch_*}` | not read by this lane; purpose unknown to me | no label from me |

**By-name references that break if a folder moves:**
- `lane_cellroute_20261109/inference/params_build.py` reads `lane_cd_20261108/reads/freeze_cd_params_20261109.json` and `freeze_cd_params_20261111.json`, and the tier params `freeze_cd_params_t{1,2,3}_20261112.json`.
- `lane_cellroute_20261109/inference/manifests/deploy*.json` lists `lane_cd_20261108/arms/{dinov3_b16_f99_s46b,dinov3_b16_f99_s47,vitl16_f99_s46,vitl16_f99_s47}/...`, `fulldata_recipeA_20261028/...`, `deploy_review_20260905/staged/model/*`, `cpsam_models/cpsam_v2`.
- Model archives use paths of the form `./experiments/lane_cd_20261108/arms/...`, and `package_20261112/*/model` trees are hard links of those paths. Built archives survive a move; rebuilds do not.
- `cd_common.py` → `meta_ceiling_20261103/source`.
- `cd_read_20261108.py` → `recipeA_analysis_20261028/hybrid_oof.json`, `lane_backbone_20261108/runs/vitl16_avg_f{0,4}_s46`, `lane_gpu3_20261106/runs`, `lane_gpu2_20261106/runs`, `lane_cd_20261108/tta`.
- `cd_stack_20261109.py` → `lane_hex_20261108/seg_loose/c*.npz`.
- `ceiling_meta_cpu_20261103.py` `C.RECIPE` / `HIST` → the recipe A and Phase I run folders above.
- CV lane `cv_tiers_20261112.py` → `lane_cd_20261108/arms/dinov3_b16_f{1,2,3}_s{46,47}` and `lane_gpu{2,3}_20261106/runs`.
- `FREEZE_CANDIDATE.md` cites all of the above.

## 6. Loose ends

- **Not verified:** cell-route tier deploy params vs a refit with no outer fold; the p / t3 image parity and T4 timing (deploy lane's own checks).
- **Strict finding the PI should know:** the full candidate's CD (recipe A e8-19 snapshot plus ViT-L e8-11) is no better than CD t1 under strict nesting (6.1743 vs 6.1737). The full image carries ~9 GB of CD-only weights and 32 forwards per image for nothing measurable.
- **HEX L5:** the HEX lane states its L5 difference as 0.0; my rerun gives ≤ 1.5e-13, which is BLAS rounding. Not a correction of substance.
- **Wording slip:** my 21:31 message to the coordinator called the anatomy "11-09/11"; it is 11-11 [D-CDANAT-001].
- **RESULTS.tsv is behind the ledger:** the 11-13 reads (O-CD-SUB1-CALIB, O-CD-PHASE1-CALIB) are in the ledger and in `reads/` but not in `RESULTS.tsv`; per the PI order no other file was written.
- **Phase I comparison basis:** the Phase I model's per-row predictions use a development checkpoint for its sixth member (not the release checkpoint), folds 2-4 only. Its calibration SCALE was fitted on all folds.
- **Processes and GPU:** none of this lane's processes run on either pod (checked 21:45 EST). The GPU 0 keepalive (pid 643028) was stopped by the coordinator [D-GPU-002]. The loop and all wake-ups of this session are cancelled.
