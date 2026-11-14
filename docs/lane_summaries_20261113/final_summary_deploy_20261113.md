# Final summary: deploy lane (cell-route step and packaging), 2026-11-13 21:55 EST

Author: deploy lane (folder `experiments/lane_cellroute_20261109`, "L" below).

## Conventions

- Strict values are five-fold values in the chain convention unless stated otherwise.
- "Strict mean" is the mean of CD, CV and HEX MAPE.
- Measured facts only. Ledger ids are in brackets.
- Nothing was deleted, moved or renamed for this summary.
- None of my processes run on the pod or the build box.
- Pod PID 599163 (`colab --auth oauth2 status -s abf17a`, running for 25 h) is not mine.

## 1. Outcome

- **Cell-route step (D-HEX-005), kept version P2 win+full on the 39-column ridge.**
  - As computed: 9.70581 vs its base chain_joint_attnp 9.71882 (-0.01301).
  - Strict on the same base: 9.71576 (-0.00306) [O-DEPLOY-006, F-DEPLOY-001].
  - End-to-end strict: 9.71342 vs HEX strict 9.71977 (-0.00635). On folds 0+4: +0.00132 [O-DEPLOY-007, A-DEPLOY-004].
  - L5 check passed:
    - The no-outer-fold refit reproduces `params/cellroute_deploy.json` exactly [O-DEPLOY-002].
    - The no-outer-fold version choice picks the kept version, with a gap of 0.00207 to the second choice [O-DEPLOY-008].
- **Cell-route on the HEX tiers, end-to-end strict:** t2 9.71943 (-0.0015), t3 9.73076 (-0.0019). Both gains are under the 0.005 noise floor [O-DEPLOY-014].
- **Tiers, strict CD / CV / HEX (mean)** [O-DEPLOY-018, D-PKG-003]:

  | tier | CD | CV | HEX | mean |
  |---|---|---|---|---|
  | full | 6.1743 | 10.1160 | 9.71342 | 8.6679 |
  | p | 6.1737 | 10.1115 | 9.71943 | 8.6682 |
  | t1 | 6.1737 | 10.1115 | 9.71342 | 8.6662 |
  | t2 | 6.1875 | 10.1115 | 9.71943 | 8.6728 |
  | t3 | 6.2138 | 10.1138 | 9.73076 | 8.6861 |

- **T4 path** (container default below compute capability 8.0: fp16 with 1 detector copy).
  - Strict-score impact, 200 labelled rows applied to the t1 strict OOF: t1 +0.0066 (+0.0016 to +0.0121), p +0.0067 [O-DEPLOY-020, O-DEPLOY-021].
  - All-fp32 gives +0.0069 [O-COORD-F32-200].
  - No path meets the 0.05 % parity rule; an A10G proxy gives +0.0059.
- **Real-T4 wall time per job (Colab, cold reads):**
  - p: 287-288 s (VM 1) and 311-314 s (VM 2).
  - t3: 173-174 s.
  - t1: 323-332 s.
  - all-fp32 and mix did not finish within 600 s.
  - [O-DEPLOY-T4-COLAB, O-COORD-COLAB-T4-B]
- **Build box (RTX 4090, Grand Challenge flags, uid 10001):** 17 of 17 jobs passed for each of p and t3 [O-DEPLOY-022, O-DEPLOY-023].
- **Packaging bugs found and fixed:**
  - PYTHONPATH not set [F-DEPLOY-002].
  - timm's `test_time_pool.py` filtered out; unloading off (out of memory); missing CV parameters gave silent constants [F-DEPLOY-003].
  - A mode-600 code file gave constants on every image; `cpsam_v2` is mode 600; t3 had no timm pin [F-DEPLOY-005].
- **Upload risk:** the p model archive is 15.06 GB. Several Grand Challenge pages state a 10 GB per-upload cap. t3's archive is 8.81 GB [F-DEPLOY-006].

## 2. Models

**Definitive file lists:** `L/package_20261112/<tier>/context/model_files.txt` for p and t3, and `L/package_20261112/context/model_files.txt` for full. Paths are relative to `/workspace/clear-ec-research` and equal to the paths under `/opt/ml/model`.

**Tier p** (443 files, 16.25 GB, 35 fp16 copies) and **tier t3** (384 files, 9.51 GB, 9 fp16 copies):

| folder (under experiments/ unless noted) | p GB / files | t3 GB / files | fp16 copies |
|---|---|---|---|
| green_track_20261028/deploy/detectors | 4.115 / 12 | 4.115 / 12 | no |
| deploy_review_20260905/staged/model (Phase-1 five) | 3.047 / 5 | 3.047 / 5 | yes (5) |
| lane_hex_20261108/deploy/detectors | 1.372 / 4 | - | no |
| cpsam_models/cpsam_v2 (top level, not under experiments) | 1.234 / 1 | 1.234 / 1 | no (mode 600 at source) |
| lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46 | 0.885 / 5 | - | yes |
| lane_cd_20261108/arms/dinov3_b16_f99_s46b | 0.858 / 5 | 0.172 / 1 | yes |
| lane_cd_20261108/arms/dinov3_b16_f99_s47 | 0.858 / 5 | - | yes |
| lane_hex_20261108/deploy/cvw3_f99_s46, cvw3_f99_s47 | 0.858 / 5 each | - | yes |
| lane_cd_20261108/arms/vitl16_f99_s46, vitl16_f99_s47 | 0.607 / 1 each | - | yes |
| fulldata_recipeA_20261028/regression_std_s46, _s47 | 0.343 / 1 each | same | no |
| lane_cv_20261108/graph_models | 0.101 / 200 | same | no |
| lane_cv_20261108/frozen_heads_full | 0.100 / 60 | same | no |
| lane_hex_20261108/deploy/graph_models | 0.063 / 125 | 0.051 / 100 | no |
| meta_ceiling_20261103/sweep/sim_pred | 0.001 / 3 | same | yes |
| lane_hex_20261108/deploy/frozen_heads | 0.000 / 4 | - | no |

- **Full tier:** 490 files, 29.98 GB. It adds 24 more fulldata_recipeA files, 8 more lane_cd files, 10 more lane_cv files and 5 more lane_hex files.
- **fp16 copies** live at `L/deploy_fp16/<original repo path>`: 82 files, 22.31 GB, each bit-identical to original → fp16 → fp32. The sha256 list is `L/deploy_fp16_sha256.txt` (82 lines) [A-DEPLOY-003].
  - The packaged entry reads the fp16 copy instead of the original for every path in `inference/docker/fp16_files.txt`.
  - The package model trees store the fp16 copy at the original path.
- **Full-data (fold99) weights per lane**, as loaded by the deploy entry:

  | lane | folder |
  |---|---|
  | CD | lane_cd_20261108/arms/{dinov3_b16_f99_s46b, dinov3_b16_f99_s47, vitl16_f99_s46, vitl16_f99_s47}/runs/fold99/ |
  | HEX | lane_hex_20261108/deploy/{cvw3_f99_s46, cvw3_f99_s47, detectors, graph_models, frozen_heads} |
  | Backbone (attnp5) | lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46/raw/runs/fold99/dinov3_b16/ (epochs 8-11 and final) |
  | CV | lane_cv_20261108/{graph_models, frozen_heads_full} |
  | Recipe A | fulldata_recipeA_20261028/regression_std_s46, regression_std_s47 |
  | Green track (detectors, hex6) | green_track_20261028/deploy/ |
  | Phase-1 | deploy_review_20260905/staged/model |

- **Upload files on the build box** (`qudgks@210.107.192.239:~/workspace/clear-ec/package_20261112/upload_20261113/`, see `SHA256SUMS.txt`) [A-DEPLOY-012, A-DEPLOY-013]:

  | file | bytes | sha256 |
  |---|---|---|
  | clear-ec_20261112-freeze-9687076-p.tar.gz (image dfc08a448e58) | 4,114,574,109 | 98f57ec5ac9fcfdcaaf880e3321e7b5a684c4f8fbacb300351ad79242a0e573f |
  | model_freeze-9687076-p_20261112.tar.gz | 15,057,667,603 | fff3ec09e155841f43057316e8398fe01ef57c5f18b97530a5daa94df43f58c5 |
  | clear-ec_20261112-freeze-9687076-t3.tar.gz (image 8853d77c1f5a) | 4,114,563,350 | 4411cb7a7b3761102200ed6cb91d4161215d1067952e08a14c0bf25e9764fbc5 |
  | model_freeze-9687076-t3_20261112.tar.gz | 8,809,046,691 | 7caaa02693776c7ca50840cfc83b94044320d788f36373a796ebb9503404415d |

## 3. Data

| path (under experiments/) | what | regenerable? |
|---|---|---|
| L/inference/params/deploy{,_p,_t1,_t2,_t3}.json | Entry parameters with the lanes' parameter files embedded (CD freeze params, HEX joint params, CV deploy folder pointer, decisions list) | yes, `L/inference/params_build.py` (TIER env), while the lane files below exist |
| L/inference/manifests/deploy{,_p,_t1,_t2,_t3}.json | Member weight paths per tier | yes, params_build.py |
| L/inference/params/cellroute_deploy{,_p,_t1,_t2,_t3}.json | Cell-route kept version refit with no outer fold on each tier's HEX OOF (p = chain_joint_t2) | yes, params_build.py with `L/source/corr_newbase.py`; inputs are the HEX tier OOF files |
| lane_hex_20261108/preds/deploy_params_joint_attnp.json, deploy_params_joint_t2.json, deploy_params_joint_t3.json | HEX ridge parameters (read by params_build) | HEX lane scripts |
| lane_cd_20261108/reads/freeze_cd_params_t1_20261112.json, _t3 (and 20261111 / 20261109) | CD stack and stretch parameters (read by params_build) | CD lane scripts |
| lane_cv_20261108/deploy_t1/{cv_deploy_params.json, cv_ridge_params.npz}, deploy_t3/... | CV stack, read at run time from the image | CV lane scripts |
| green_track_20261028/deploy/hex6_blend_params.json | hex6 blend parameters, read at run time | green track; not re-derived by me |
| lane_gpu2_20261106/frozen_s46/pca.npz | PCA basis opened by a member module at run time (census) | origin lane_gpu2; not re-derived by me |
| phase2_foundation_20261002/weights/dinov3-vitb16/config.json | DINOv3-B config (base weights are not shipped) | copy of the HF config |
| L/inference/docker/base_shapes.json | Tensor shapes and sha256 of the skipped base weights (timm ViT-L 318, CPDINO 191) | yes, docker/trace_base_open.py |
| L/inference/docker/fp16_files.txt | 82 paths served from deploy_fp16 | yes, docker/make_pack_lists.py |
| L/inference/docker/pack_census{,_p,_t1,_t2,_t3}.json | Files and modules each tier opens; input of stage_package.py | yes, inference/test_packaged.py (GPU run) |
| L/out/corr_hex_*.npz / .json | Cell-route OOF, strict and end-to-end strict predictions | yes, `L/source/corr_newbase.py`, corr_strict.py, corr_strict_e2e.py |
| L/out/t4_*.json, colab_*.json, box_test_{p,t3}_20261113.json | T4-path, Colab and build-box measurements | test_t4.py, compare_t4.py, impact_t4.py, impact200.py, colab_compare.py, buildbox_20261113/compare_box.py; Colab outputs are not reproducible (VMs gone) |
| lane_cd_20261108/reads/cd_tier_t1_20261112.npz, lane_cv_20261108/deploy_t1/cv_final_oof.npz, L/out/corr_hex_e2e_strict_a_chain_convention.npz | Strict t1 OOF used by impact_t4.py (sanity values 6.1737 / 10.1115 / 9.71342) | lane scripts |

**Test-only inputs** (excluded from the packages):
- `datasets/clear_ec_fullframe_20260904/img_full_raw.dat` (memmap, 8954×972×1296 uint8)
- `lane_cd_20261108/reads/freeze_cd_oof_20261111.npz` (row index)
- `L/inference/rows200_20261112.npy` (make_rows200.py)
- `L/inference/colab/frames/*.mha` (5 frames, make_colab_frames.py)
- The platform debug cases `0001-19/20/21.mha` (build box `~/workspace/clear-ec/guard_test_20261031/cases`)

## 4. Ideas

**(a) Adopted**
- `cellroute-kept-version-refit-no-outer-fold`: HEX end-to-end -0.00635 on t1 [D-HEX-005, O-DEPLOY-007].
- `fp16-storage-regression-members-upcast-at-load`: parity median ≤ 0.009 % [O-DEPLOY-005, O-DEPLOY-012].
- `skip-base-weights-load-from-config`, `single-cpsam_v2-copy`: -1.90 GB and -1.23 GB [A-DEPLOY-002].
- `per-stage-model-unloading`: GPU peak 17.35 → 4.33 GB [O-DEPLOY-015].
- `packaging-tiers-p-t1-t2-t3`: build argument TIER [A-DEPLOY-007].
- `t4-default-fp16-one-detector-copy`: impact +0.0066; one copy changes outputs by < 5e-7 [O-DEPLOY-020].
- `non-root-readability-check-at-build`: chmod plus a find check as uid 10001 [F-DEPLOY-005].
- `model-archive-world-readable`: tar `--mode=a+rX` [F-DEPLOY-005].

**(b) Closed**
- `fp16-every-tensor-storage`: fails parity (CV 0.168 %, HEX 0.381 % median) [O-DEPLOY-005].
- `t4-all-fp32-path`: impact +0.0069 vs fp16's +0.0066; > 600 s on the real T4 [O-COORD-F32-200, O-DEPLOY-T4-COLAB].
- `t4-mixed-fp32-detectors`: +0.0053, > 600 s; mixhex6 364.9 s [O-DEPLOY-020, O-COORD-COLAB-T4-B].
- `four-detector-copies-on-t4`: 376-398 s vs 323-332 s on t1, same score [O-DEPLOY-T4-COLAB].
- `parity-0.05pct-off-h200`: unattainable; every non-H200 kernel moves HEX by about 0.4 % median, A10G proxy +0.0059 [O-DEPLOY-020].
- `cellroute-on-t2-t3-as-a-gain`: -0.0015 / -0.0019, under the noise floor. The step is still in p [O-DEPLOY-014].

**(c) Open or partly tested**
- `fourier-ring-cd-column`: alone, nested MAPE 9.59 (best variant). Its L1 read showed the column's deviation from the stack does not predict the stack error (r 0.021). The CD stack read (Q-DEPLOY-001) was not seen by me [O-DEPLOY-010, O-DEPLOY-011].
- `p-model-archive-under-10GB`: move about 5.5-6.3 GB of weights into the image, or add more fp16 copies (needs parity reruns). Not tested [F-DEPLOY-006].
- `platform-t4-exec-time-p`: 287-314 s on two Colab VMs; 72-80 s on the 4090. The platform's Try-out decides; the 270 s rule applies [A-DEPLOY-014].

## 5. Dependency closure and folder map

### (i) Folders a rebuild needs by name (must not move or disappear)

**Code** (referenced by the staged contexts, `stage_package.py`, the manifests and params, and `gc_entry.py`):
- `experiments/lane_cellroute_20261109/{inference, source, deploy_fp16, package_20261112}`
- `experiments/phase2_foundation_20261002/runtime/{dinov3, timm, dinov3-0.0.1.dist-info, timm-1.0.20.dist-info}` and `.../weights/dinov3-vitb16/config.json`
- `experiments/{eval_harness_20261031, green_track_20261028, lane_backbone_20261108, lane_cd_20261108, lane_cv_20261108, lane_gpu2_20261106, lane_hex_20261108, meta_ceiling_20261103, oof_recipeA_20261027, phase2_crossfold_20261015, submission1_test_20261030}`
  - These hold the .py modules (the context copies them; `phase2_crossfold_20261015/source/phase2_lejepa_20261003.py` is the mode-600 file).
  - They also hold the params, the CV deploy folders and `pca.npz`.

**Weights:** every path in `package_20261112/context/model_files.txt` (full tier, a superset of p and t3):
- `cpsam_models/`
- `experiments/deploy_review_20260905/staged/model`
- `experiments/fulldata_recipeA_20261028`
- `experiments/green_track_20261028/deploy`
- `experiments/lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46`
- `experiments/lane_cd_20261108/arms/*_f99_*`
- `experiments/lane_cv_20261108/{graph_models, frozen_heads_full}`
- `experiments/lane_hex_20261108/deploy`
- `experiments/meta_ceiling_20261103/sweep/sim_pred`
- `experiments/lane_cellroute_20261109/deploy_fp16`

**Params rebuild only:**
- `lane_hex_20261108/preds`
- `lane_cd_20261108/reads`
- `lane_cv_20261108/deploy_t1`, `deploy_t3`
- `L/out/corr_hex_*`
- The labels and OOF read by `params_build.py` and `corr_newbase.py`

**Tests only:** `datasets/clear_ec_fullframe_20260904`, `lane_cd_20261108/reads/freeze_cd_oof_20261111.npz`.

### (ii) Folder map

**Pod: `L/package_20261112/`**
- Every tier's `model/` is hard links. `cpsam_v2` has link count 6 (the original plus the full, p, t1, t2 and t3 trees). They are not copies, so deleting a tier's `model/` frees nothing while the originals exist.
- Every `context/` is a real copy (about 9 MB, 564-567 files).

| path | purpose | class | reason |
|---|---|---|---|
| L/package_20261112/{context, p/context, t1/context, t2/context, t3/context} | Build contexts (A-DEPLOY-011 hashes) | KEEP | Needed for any rebuild; tiny |
| L/package_20261112/p/model, t3/model | Hard-linked archive trees of the uploadable tiers | KEEP-MODEL | Source for re-archiving p / t3 |
| L/package_20261112/model (full), t1/model, t2/model | Hard-linked trees of tiers not uploaded | DELETE-AFTER-SUBMISSION | No space freed; they keep data alive if originals go |
| L/package_20261112/UPLOAD_TRYOUT_KO.md, report_append_20261113.md | Note copy; REPORT append source | KEEP | Small |
| L/deploy_fp16 | 82 fp16 copies, 22.31 GB; only copy in that format; read by every package | KEEP-MODEL | Regenerable by docker/convert_fp16.py, but the sha list is the check |
| L/inference (incl. docker/, params/, manifests/, buildbox_20261113/) | Entry, packaging, tier params, box scripts | KEEP | Rebuild source |
| L/inference/colab (incl. b/, frames/, diag/) | Colab real-T4 scripts and 5 test frames | ARCHIVE | The Colab VMs are gone; the frames are also used by the box test |
| L/source, L/out | Cell-route fits; all measurement outputs | KEEP | Ledger sources |

**Build box: `~/workspace/clear-ec/`**

| path | purpose | class | reason |
|---|---|---|---|
| package_20261112/upload_20261113/{4 tar.gz, SHA256SUMS*, UPLOAD_TRYOUT_KO.md} | Upload files | KEEP | Until accepted; then keep the submitted pair |
| package_20261112/upload_20261113/stale_perm600_20261113/ | First p image archive (constants under uid 10001), 4.11 GB | DELETE-NOW | Broken; superseded |
| package_20261112/{p,t3}/{context,model} | rsync copies of the pod trees (not links), 16.25 + 9.51 GB | DELETE-AFTER-SUBMISSION | Needed only for a rebuild on the box |
| package_20261112/t1/context | Stale partial copy from the failed stream | DELETE-NOW | Not a valid context |
| package_20261112/test_model_p, test_model_t3 | Extracted archives used by the container tests, 16.25 + 9.51 GB | DELETE-AFTER-SUBMISSION | Retest only |
| package_20261112/{test_20261113, frames_colab5} | Container-test job folders; copies of the 5 pod frames | DELETE-AFTER-SUBMISSION | Results are on the pod (out/box_test_*.json) |
| package_20261112/{scripts_20261113, logs} | Box scripts (copies on the pod); build, pack and test logs (box only) | KEEP | Small; logs are evidence |
| docker images clear-ec:20261112-freeze-9687076-{p,t3} | Built images | KEEP | Until submission |
| dangling images b982cbddf583 and the failed t3 build layers | Superseded | DELETE-NOW | Constants bug / failed build |
| package_20261030, 20261031, 20261101, 20261104; submission_hybrid_*; final_test_20261104; sub2_test_20261101; container_test_0234-20; guard_test_20261031; model/ | Submission 1 and 2 archives and tests (`model_hybrid_20261030.tar.gz`, 9,210,772,470 B, appears in three folders) | ARCHIVE | Not my artifacts; keep one copy of each submitted pair (coordinator's records). guard_test_20261031/cases holds the debug cases |

**Pod: `/root` leftovers**

| path | purpose | class | reason |
|---|---|---|---|
| /root/.ssh/buildbox_20261112(.pub) | The PI's key for pod → box rsync | DELETE-AFTER-SUBMISSION | May be needed for t3 or a rebuild |
| /root/.ssh/config (hosts colabt4, colabt4b) | Colab ssh entries | DELETE-NOW | Colab VMs gone |
| /root/bin/cloudflared | Tunnel for `ssh colabt4` | DELETE-NOW | Colab only |
| /usr/local/bin/colab, ~/.config/colab-cli, ~/.config/python_keyring | Colab CLI and the PI's saved OAuth token (O-COORD-COLAB-T4: to be removed after the test) | DELETE-NOW | Token should not stay on a shared pod. Stop PID 599163 first (not mine) |

## 6. Loose ends

**Unverified**
- `cellroute_deploy_{p,t2,t3}.json` were never checked (L5) against a separate no-outer-fold refit; only the full / t1 file was (O-DEPLOY-002, O-DEPLOY-008). The CD lane did not check them either.
- The platform T4 exec time for p is unmeasured. Colab gave 287-314 s.
- The 10 GB upload cap is unverified (F-DEPLOY-006).
- The Colab t3 run used a copy of the t1 CV folder, so its CV comparison is not like for like (O-DEPLOY-023).

**Corrections to earlier ledger lines**
- A-DEPLOY-010's tree hashes and Dockerfile sha are superseded by A-DEPLOY-011 (p 484c0d77e349502c, t3 d9e48922389c06ae, Dockerfile 2daeb1a20da917ce).
- F-DEPLOY-004 and A-DEPLOY-010 treated the contexts as ready. They still had the uid-10001 permission bug.
- Every pod and Colab test (O-DEPLOY-015/017, O-DEPLOY-T4-COLAB, O-COORD-COLAB-T4-B) ran as root, so none of them covers file permissions.
- Image b982cbddf583 (O-COORD-BUILD-P) is superseded.
- O-DEPLOY-013 ("T4 not viable") and D-PKG-003 ("A10G required") predate unloading and the T4 path. They are superseded by D-PKG-006 and O-DEPLOY-T4-COLAB.

**What a rebuild from scratch needs**
1. Every path in model_files.txt (full) at its repo path, `L/deploy_fp16` and the code folders of 5(i).
2. Either the staged contexts, or the stage inputs: `stage_package.py`, `pack_census_<tier>.json`, `fp16_files.txt`, `base_shapes.json` and `Dockerfile.freeze`.
3. Base image `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime@sha256:3bb77138e105723dd4ed760b82fb63d8310ae3a1afc58f76e0ecf0f776568d33`.
4. PyPI access for the pinned requirements.
5. Build with `docker build --build-arg TIER=<tier> -t clear-ec:20261112-freeze-9687076-<tier>`.
6. Archive with `inference/buildbox_20261113/pack_tier_20261113.sh` (it sets `--mode=a+rX`).
7. Test with `test_tier_20261113.sh`, which runs as uid 10001 against the extracted archive.

**Checked 22:15 EST (PI request 21:55; read-only on the build box, no rebuild, nothing deleted)** [O-DEPLOY-024]

1. **CV folder each image reads.** I instantiated `recipe_infer.Recipe` inside each image (uid 10001, extracted model mounted) and printed the parameters it actually loaded.
   - **t3** (8853d77c1f5a): reads `experiments/lane_cv_20261108/deploy_t3`, loaded weight fz:autocorr 0.125.
   - **p** (dfc08a448e58): reads `experiments/lane_cv_20261108/deploy_t1`, loaded weight fz:autocorr 0.25.
   - The other five weights are identical in both (dino 0.125, fz:spectrum 0.25, fz:cov_sqrt 0.375, graph:plain 1.0, graph:order 0.25).
   - sha256 prefixes inside the images equal the pod's files:

     | folder | cv_deploy_params.json | cv_ridge_params.npz |
     |---|---|---|
     | deploy_t3 | fad345ca7428cc87 | 744b16d26998e765 |
     | deploy_t1 | ad9cec5d40371d8c | 744b16d26998e765 |

   - The ridge file is identical in t1 and t3.
2. **The 11-13 t3 box test used the t3 CV stack.** All 17 jobs in `test_20261113/t3/*/*/stdout.json` have decisions `'D-PKG-002 tier t3'` and `'CV lane_cv_20261108/deploy_t3'`. These are written into the same `params/deploy.json` whose `cv.deploy_dir` the entry loads, and that file is unchanged in the image since the test. A rerun was not needed. The Recipe check in point 1 confirms the loaded weight 0.125.
3. **The upload image archives were saved from exactly these images.** The `Config` in `manifest.json` inside `clear-ec_20261112-freeze-9687076-p.tar.gz` is `sha256:dfc08a448e58...` (RepoTags `...-p`), and inside the t3 archive it is `sha256:8853d77c1f5a...` (RepoTags `...-t3`). Both equal the image ids.

The PI said `du` does not matter: the CV lane's self-report needs no follow-up.
