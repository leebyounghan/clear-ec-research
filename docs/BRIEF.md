# CLEAR-EC Phase II brief

Measurements only. No experiment history and no verdicts on methods, on purpose: draft ideas from this page first, then check them against the ledger (last section).

## Task

| Item | Value |
|---|---|
| Input and output | One specular image, 1296 x 972 px, 1.0 x 0.75 mm, 0.7716 um/px; predict CD, CV, HEX |
| Score | Mean absolute percentage error, equal weight on the three metrics; zero labels not scored |
| Test | 1000 hidden images, two submissions, best counts; failed evaluations do not count. Submission 2 unused |
| Deadline | 2026-11-15 06:00 Toronto |
| Runtime | 5 minutes per case, 32 GB RAM, T4 or A10G |
| Data | 8953 labelled frames; 25 frames with expert cell markers, all one style (2023); external Topcon set, 385 images with cell points |
| Constraint | No human annotation |

## Leaderboard, 2026-11-07

| Rank | Entry | Score | CD | CV | HEX |
|---|---|---:|---:|---:|---:|
| 1 | V14 | 8.6920 | 6.049 | 9.928 | 10.099 |
| 2 | LKSOM 2 | 8.7805 | 6.089 | 10.068 | 10.184 |
| 3 | ec blend | 8.7816 | 6.110 | 10.029 | 10.206 |
| 4 | Ours, submission 1 | 8.9675 | 6.250 | 10.143 | 10.510 |
| 5 | first submission | 37.98 | | | |

Their methods are not visible: algorithm pages need sign-in and a permission request.

## Our position

Out-of-fold (OOF) over 8953 frames. "Projected" adds the OOF change to the submission 1 test score: arithmetic, not a forecast.

| Metric | Submission 1 OOF | Submission 1 test | Test minus OOF | Best OOF | Change | Projected test | Gap to V14 | Gap to LKSOM 2 | Gap to ec blend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CD | 6.312 | 6.250 | -0.062 | 6.264 | -0.048 | 6.202 | +0.153 | +0.113 | +0.092 |
| CV | 10.204 | 10.143 | -0.061 | 10.151 | -0.053 | 10.090 | +0.162 | +0.022 | +0.061 |
| HEX | 10.115 | 10.510 | +0.395 | 9.803 | -0.312 | 10.198 | +0.099 | +0.014 | -0.008 |
| Mean | 8.877 | 8.9675 | +0.090 | 8.739 | -0.138 | 8.83 | +0.138 | +0.05 | +0.05 |

Paired differences on 8953 rows have a CI of about +-0.02 on the mean; on 1000 test rows, roughly three times that.

Best pipeline, not yet in an image: hybrid B (0.5 recipe A + 0.5 crop CPSAM members), hex6 on HEX, log-affine stretch, cell-graph network on CV and HEX (Voronoi graph of pre-only detector peaks in the three windows with most confident peaks, three detectors), ridge on a graph encoder pretrained on simulated tessellations. Submission 1 is hybrid B alone.

The table above is the position of 2026-11-07 and is kept as the baseline. The current candidate is in the next section.

## Freeze 2026-11-11 18:00 EST

Source: the shared ledger `ledger_20261111.jsonl` (record ids in brackets) and the lane notes in the same folder. Protocol: [FRAMEWORK_REVISION_20261111.md](FRAMEWORK_REVISION_20261111.md) and the section "Shared ledger, decision ids, lessons, verifier" of [LANES_20261108.md](LANES_20261108.md).

All values are five-fold OOF MAPE unless a fold set is named. "As computed" is the lane's nested stack as built. "Strict" hides the scored fold's labels in the whole procedure, including every choice (lesson L4). The candidate is not yet in an image and has no test score.

### Candidate per metric

| Metric | Decision | Candidate | As computed | Strict | Best pipeline, as computed | Best pipeline, scalars refit per outer fold |
|---|---|---|---:|---:|---:|---:|
| CD | D-CD-001 | P1 x SCALE / recipe A s46+s47 snapshot / DINOv3-B s46+s47 seed mean TTA / ViT-L s46+s47, plus mask density cd_mm2, then stretch | 6.17413 | 6.17428 | 6.26387 | 6.26103 |
| CV | D-CV-001 | 6-component stack without the order-loss DINOv3 (3-seed rule) | 10.11014 | 10.11596 | 10.15119 | 10.14786 |
| HEX | D-HEX-004 | 39-column joint nested ridge with the attention-pool column | 9.71882 | 9.71977 | 9.80262 | 9.79992 |
| HEX with the cell-route step | D-HEX-004 + D-HEX-005 | Kept feature version (P2 win+full) on the 39-column ridge | 9.70581 | 9.71342 | | |

- CD and CV values are the verifier's recomputation from each lane's OOF file [O-VER-001]. The decisions quote them as 6.1741 / 6.1743 and 10.1101 / 10.1160.
- HEX, D-HEX-004 [O-HEX-001, A-HEX-002]: strict per fold 10.0362 / 9.4574 / 9.8653 / 9.5799 / 9.6602; strict on folds 0+4 9.84826 against the best pipeline 9.90790. Chain convention (base fitted on the fit rows). D-HEX-004 supersedes the chain (D-HEX-001), the 38-column ridge (D-HEX-002) and the reverted chain (D-HEX-003).
- HEX fallbacks [O-HEX-001, A-HEX-001]: fallback 1, the 38-column joint ridge, 9.72621 nested / 9.72662 strict (folds 0+4 strict 9.85123). Fallback 2, the 10-step chain, 9.73258 as computed, 9.74062 strict for the fixed composition, 9.74961 strict for the chain procedure (folds 0+4 9.87964).
- Cell-route step, D-HEX-005 [O-DEPLOY-006, O-DEPLOY-007, F-DEPLOY-001]:
  - On the nested 39-column OOF (9.71882) as the base on every row: 9.70581 as computed (-0.01301), 9.71576 strict for the step's own choices (-0.00306). Folds 0+4: 9.84236 (-0.00589) and 9.85371 (+0.00546). Without the 25 gate frames: as computed -0.0073, strict +0.0021.
  - End to end strict, on the strict 39-column OOF (9.71977, recomputed in the same run): chain convention 9.71342 (-0.00635), per fold -0.0044 / -0.0160 / -0.0206 / +0.0022 / +0.0071, folds 0+4 9.84958 (+0.00132), without the 25 gate frames -0.00356. OOF convention (deploy-like base) 9.71663 (-0.00314), folds 0+4 9.85366 (+0.00541), without the gate frames +0.00211. The kept version with no version choice on the same two bases: 9.70727 and 9.70679.

### Integrated table (verifier)

The CD lane, which authored neither the CV nor the HEX stack, recomputed every lane's claim from that lane's own OOF file: no difference above 0.0005 [O-VER-001]. Mean of the three metric means, 8,954 rows, HEX scored on 8,952 [O-VER-002, O-VER-004].

| Row | CD | CV | HEX | Mean |
|---|---:|---:|---:|---:|
| Candidate, as computed | 6.17413 | 10.11014 | 9.70581 | 8.66336 |
| Candidate, strict, cell-route step strict on the nested HEX base [O-VER-002] | 6.17428 | 10.11596 | 9.71576 | 8.66867 |
| Candidate, strict, HEX end to end, chain convention [O-VER-004] | 6.17428 | 10.11596 | 9.71342 | 8.66789 |
| Best pipeline, as computed | 6.26387 | 10.15119 | 9.80262 | 8.73923 |
| Best pipeline, P1 SCALE and recipe A scalars refit per outer fold | 6.26103 | 10.14786 | 9.79992 | 8.73627 |

- Strict against strict: 8.66789 against 8.73627, -0.0684 [O-VER-004]. Relayed to the PI as CD 6.1743, CV 10.1160, HEX 9.7134, mean 8.6679, -0.068 against 8.7363 [O-VER-003].
- Mean on the 8,952 common rows: candidate 8.60020 as computed / 8.60525 strict (O-VER-002 row), best pipeline 8.67397 / 8.67096 [O-VER-002].
- The verifier assembled the strict HEX 9.71977 independently from the HEX lane's fold runs and found it identical (max abs 0.0) to the lane's file `preds/chain_joint_attnp_strict.npz` [O-VER-004, O-VER-005].
- One backbone-lane line quotes "integrated candidate 8.6736 as computed" [O-BB-SYNFT3-5F]. It does not match the verifier's 8.66336; the verifier's value is the one used here.

### Deploy equals evaluated (lesson L5)

- CD: passes up to a uniform 7.2e-5 scale. The strict code with no outer fold refits the constants (P1 0.99743, recipe A 0.98921, stretch_c 0.99791, mu 7.91741); weights and mask are identical; in-sample CD 6.16740 against 6.16733 [F-VER-001].
- CV: the nested stack code with no outer fold reproduces `deploy/cv_deploy_params.json` (md5 3b437500) exactly, weights [0.125, 0.25, 0.375, 0.25, 1.0, 0.25]; stretch max abs diff 1.1e-16 [O-CV-L5-001]. The verifier's rerun gives the same weights [F-VER-001].
- HEX ridge: the nested procedure with no outer fold against the stored deploy parameters, max diff 0.0 (deploy: base ap1, alpha 1, stretch c 0.99561, g 1.025, mu -0.61800) [A-HEX-001]. Checked by the HEX lane; the verifier cited it and did not rerun it [F-VER-001].
- Cell-route step: the fold fit with no outer fold reproduces `params/cellroute_deploy.json` exactly (max abs diff 0.0, 8825 training rows) [O-DEPLOY-002]. The strict version choice with no outer fold picks the deployed version P2 win+full (9.70581; second 9.70788, gap 0.00207) [O-DEPLOY-008]; the verifier's rerun agrees [F-VER-002].
- Result: L5 passes for all four parts, CD up to a uniform 7e-5 scale [F-VER-002].

### Deploy facts

- Entry: `lane_cellroute_20261109/inference/recipe_infer.py`, `Recipe(manifests/deploy.json, params/deploy.json)`, implements D-CD-001, D-CV-001, D-HEX-004 and D-HEX-005; `HEX_FORM=joint` / `chain` rebuild the two HEX fallbacks [A-DEPLOY-001]. Decision conformance of the parameter set: all checks true [O-DEPLOY-001].
- Size: 779 files, 55.42 GB fp32. About 30.0 GB with the base weights dropped (-1.90), the duplicate `cpsam_v2` removed (-1.23) and fp16 copies (-22.31) [A-DEPLOY-002, Q-001].
- Runtime on shared GPU 3 of the research pod: steady 98.1 s per frame, first frame 166 s; the largest stage is the HEX graph nets, 21.3 s for 125 nets [A-DEPLOY-002]. The 20-frame run below read 97.0 s median per frame [O-DEPLOY-004].
- fp16 copies: 82 files including the 5 attention-pool checkpoints, 44.62 GB fp32 to 22.31 GB, each bit-identical to original -> fp16 -> fp32; nothing overwritten. The entry does not read them yet [A-DEPLOY-003].
- fp16 parity, 20 deploy frames, every regression member fp16 and the detectors, hex6, `cpsam_v2`, recipe A e20 and graph models fp32, against fp32 in the same code, median / max change: CD 0.009 / 0.027 %, CV 0.005 / 0.021 %, HEX 0.007 / 0.026 %. Rule: median below 0.05 % per column; pass. fp16 for every tensor fails (median CV 0.168 %, HEX 0.381 %, measured on the earlier stacks) [O-DEPLOY-005].
- Fold-0 combine stages on stored inputs, all 1791 fold-0 rows, max relative difference to the stored OOF: CD 0, CV 5.6e-16, HEX ridge 4.4e-16, HEX after the cell-route step 4.4e-16 [O-DEPLOY-003].
- Deploy run with every member live and full-data fits, 20 training frames, |deploy / OOF - 1| median / max: CD 0.51 / 2.4 %, CV 1.32 / 5.7 %, HEX 1.63 / 4.6 %. The full-data models saw these frames, so this is not a parity number [O-DEPLOY-004].
- Attention-pool full-data build (fold 99 split, seed 46): strict load, held-row recompute max relative difference 0.0, training-row r against the OOF members 0.987 / 0.985 / 0.986 [A-BB-ATTNP-F99]; in the entry sha256 5 of 5 match and per-checkpoint HEX is within 0.25 % of the stored held-row predictions [O-DEPLOY-004].
- HEX freeze file hashes: `preds/deploy_params_joint_attnp.json` sha256 1fbc5384..., `preds/chain_joint_attnp.npz` sha256 645d0b7f... [A-HEX-004].

### Seed audit (lesson L3)

- CV: the order-loss DINOv3 component read -0.0460 with seed 46 alone (stack 10.1052) and -0.0369 with the mean of seeds 46, 47, 49 (stack 10.1143), both against the best pipeline 10.1512. 10.1143 does not beat the 6-component 10.1101, so the component is out and 10.1052 is withdrawn [L3, D-CV-001, CV lane notes].
- HEX chain: the cvw3 step read +0.0027 with seed 46 and -0.0029 with seed 47 [L3]. The chain was rebuilt with cvw3 as the mean of seeds 46 and 47 and without the `hex_pre46_s5` graph step: 9.73258 five folds, 9.86437 folds 0+4, in place of 9.7358 (HEX and cell-route lane notes, 11-11 06:35).
- CD: the DINOv3 seed mean replaces seed 46 alone, +0.0016; seed 46 was lucky [O-CD-001].
- HEX, CPSAM multi-target column (mtcnn5) second seed, inside the 39-column ridge, nested, five folds, seed 47 on folds 0+4 rows only: no column 9.71991, seed 46 9.71882, seed 47 9.71846, seed mean 9.71827; column gain -0.00109 / -0.00145 / -0.00164. Deploy unchanged, one full-data seed [O-HEX-004]. On the fallback chain, folds 0+4: without 9.87131, seed 46 9.86797, seed 47 9.86996, seed mean 9.86711 [O-HEX-005].
- Seed noise is about 0.005 [L7].

### What strict nesting changed (lesson L4)

- Choosing on reported rows was worth 0.006 (CV inner stretch), 0.010 (HEX chain step selection) and 0.010 (cell-route feature-version choice) [L4].
- CV 10.11014 to 10.11596. HEX chain 9.73258 to 9.74961. Cell-route step -0.01301 to -0.00306 on the same base.
- The HEX joint ridge read 9.7266 nested under the chain convention and 9.7414 under the OOF-input convention its first deploy fit used [L5].

### Test gap (F-TESTGAP-001)

Submission 1's test-minus-OOF pattern (CD -0.062, CV -0.061, HEX +0.395) is inside 1000-image sampling noise: joint tail 0.40; HEX alone 6.4 %. SD of the 1000-row mean gap: CD 0.318, CV 0.518, HEX 0.257. No HEX adjustment is applied: an offset fix has negative expected value when its sign is unknown. Source: `lane_cv_20261108/reads/hex_test_gap_20261111.md`.

### Source shift (F-SHIFT-001)

Leave-one-source-out: all three metrics lose across sources, and 85-90 % of the loss is a flat level offset. Penalties in MAPE points, as recorded: CD +3.4 / +3.7, CV +6.0 / +6.5, HEX +3.5 / +0.8. The mixed-source candidate has an offset of about 0 by source. Shading correction raised the offset (0.0825 to 0.0959, backbone lane notes). Test-time alignment was not started. Source: `lane_backbone_20261108/reads/shift_step1_raw.json`.

### Directions closed, 2026-11-10 afternoon to 2026-11-11

One measured line each, as recorded in the ledger or the lane notes. "Not recorded" means the value is in neither.

- Order-loss DINOv3 as a CV stack component: -0.0369 with the 3-seed mean (10.1143), does not beat 10.1101; not adopted [L3, D-CV-001].
- Order-loss ViT-L, order-loss DINOv2, order-loss graph sets, SD-target member, LDS member, Vigueras-Guillen segmentation features, on top of the CV stack: "zero or worse" (CV lane notes, 11-11 03:15); per-try values not recorded.
- C-Mixup DINOv3-B: member on folds 0+4 mean 9.233 against the plain DINOv3-B 9.2485; stack weight 0 in CD, CV and HEX (backbone lane notes; L1).
- Attention-pool DINOv3-B: member on folds 0,1,2,4 mean 9.214 against 9.219; CD +0.002 to +0.005 against 6.1741 [O-CD-001]; weight 0 in CD and CV. Kept in HEX only, as the 39th ridge column (strict 9.72662 to 9.71977).
- Shading correction: raised the source offset 0.0825 to 0.0959; closed. Appearance augmentation against the HEX gap: closed with it, never run. Test-time adaptation: not started (backbone lane notes, 11-11 05:15).
- Ensemble distillation: closed, recorded as "44 % of the gap" (backbone lane notes); which gap is not recorded. OOF teachers make a student's fold read optimistic by about 0.08-0.13 MAPE per metric (cell-route lane notes, 11-09).
- Synthetic zero-shot columns (DINOv3-B pretrained on 49k synthetic frames, no real label; label r on logs CD 0.609, CV 0.417, HEX 0.426): HEX strict 9.72086 against 9.71977 (+0.00109), CV corrector +0.0152 and member weight 0, CD weight 0 [O-BB-SYN-ZS, O-HEX-002, O-CD-001].
- Synthetic-pretrained fine-tuned DINOv3-B (synthft3), D-SYN-001, closed: single member, five folds, seed 46, 9.1606 against the plain DINOv3-B 9.2326 (-0.072) [O-BB-SYNFT3-5F]. In stacks: CV in place of the plain DINOv3 term 10.1195 against 10.1160 strict (+0.0035), as a 7th component +0.0034 [O-CV-SYN-002]; the folds 0+4 screen had read -0.0110 [O-CV-SYN-001]. HEX strict 9.72078 against 9.71977 (+0.00101), member on folds 0+4 only [O-HEX-003]. CD weight 0 on folds 0+4 [O-CD-001].
- From-scratch combination search over 22 members, CD: forward selection 6.1733 (-0.0008, folds mixed), ensemble 6.1767 (+0.0026), against 6.1741; does not beat the candidate [O-CD-001].
- HEX chain as the freeze form: superseded by the joint ridge (chain procedure strict 9.74961 against 9.71977) [D-HEX-004].
- Technician-hypothesis selection: not recorded in the ledger or the lane notes.

### Earlier numbers this section supersedes

- "Best OOF" in the "Our position" table (CD 6.264, CV 10.151, HEX 9.803, mean 8.739) is the best pipeline. It stays as the baseline; the candidate above replaces it as the current best. A projected test value for the candidate is not recorded.
- CV 10.1052 (-0.0460, order-loss DINOv3 seed 46 alone): withdrawn. CV 10.114 (-0.037, 11-09) and CV 10.111 (-0.040): superseded by 10.1101.
- CD 6.1775 (11-09): superseded by 6.1741.
- HEX chain values 9.7380 (11-09), 9.7358 (`chain_mtcnn5fix`) and 9.73258 (`chain_seedaudit10`, now fallback 2): superseded by the 39-column ridge. Cell-route step values on those bases, 9.7315, 9.7283, 9.72623, and 9.71154 on the 38-column ridge: superseded by 9.70581 / 9.71342.
- Deploy sizes 48.3 GB, 51.9 GB, 28.2 GB and 27.3 GB (cell-route lane notes, 11-10 and 11-11 morning): superseded by 55.42 GB fp32 and about 30.0 GB.

### Still running after the freeze, not part of it

- D-SYN-002: synthetic pretraining applied to ViT-L s46, folds 0 and 4 (same 50k v3 frames, then the unchanged recipe). Fold 0 landed 17:50 EST: single member 9.2039 against the plain ViT-L 9.2962 (-0.092; CD 6.445 / 6.516, CV 10.413 / 10.452, HEX 10.754 / 10.921), one fold, one seed [A-BB-SYNVITL-F0]. Fold 4 lands about 18:25 EST. Stack reads (in place of the plain member, strict, folds 0+4) are armed and not recorded yet. Kill rule: no strict stack gain on folds 0+4 in CD or CV closes synthetic pretraining for stacks. A positive read goes to the PI before any five-fold or full-data run.
- D-SYN-003: the same for recipe A s46, folds 0 and 4, on GPUs 0 and 3. Fine-tunes end about 03:00-03:30 EST 11-12 [O-BB-SYN3-ETA2]. The CD, CV and HEX reads are armed [A-CD-SWAPA-PLAN, A-HEX-003, A-CV-SYN3-READ]. No value recorded yet.

### Open for the PI

- The challenge's container / model size limit and its per-job time limit [Q-001]. Measured: 55.4 GB fp32, about 30.0 GB with the lossless savings; 98 s per frame steady, 166 s first frame on a shared GPU.
- Approval of the docker build. Pointing the entry at the fp16 copies, skipping the base weights and the build itself all wait for it.
- Whether to use the second submission.

## Measured facts

Labels:

- CD = 1e6 / integer mean cell area. Label / marker density 0.97-1.43.
- CV has 78 distinct values, step 0.01. Device CV is about twice Voronoi area CV of the same markers.
- HEX matches the six-neighbour share of markers within 1-2 % on clean sets.
- Two ID formats mark two sources (image classifier AUC 0.998); CD error 7.11 on the old one, 5.50 on the new one.
- Label of fellow eyes correlates 0.43 on CV, 0.25 on HEX.
- 2023-24 images are darker and vignetted, with higher CV and lower HEX.

The region a technician marks, 25 sets:

- Markers per image: median 148, p10-p90 119-158, CV 0.11. Hull area correlates -0.93 with CD. The recorded box is fixed, 538 x 408 px.
- The region is about 17 % of the frame. The window with most peaks >= 0.5 lies mostly inside the box (IoU 0.42, 0.52 is the most this window size allows).

Reading device values from expert markers, 25 sets: HEX r 0.84, CV r 0.88-0.90, CD r 0.75. Hybrid B on the same frames: HEX 0.81, CV 0.53, CD 0.77.

CV from expert markers with errors added on purpose, 25 sets, r and leave-one-out MAPE after scaling:

| Added error | Area CV | Neighbour-distance CV |
|---|---:|---:|
| None | 0.89, 8.8 | 0.89, 10.9 |
| Miss 11.5 % | 0.74, 13.1 | 0.80, 13.1 |
| Miss 20 % | 0.62, 15.2 | 0.77, 13.7 |
| False peaks 14 % | 0.80, 12.1 | 0.84, 12.3 |
| Jitter 10 % of spacing | 0.88, 8.6 | 0.87, 11.8 |
| Jitter 20 % of spacing | 0.83, 10.9 | 0.79, 13.8 |
| Region 70 % of the cells | 0.82, 11.9 | 0.78, 12.3 |
| Region 50 % of the cells | 0.75, 14.4 | 0.69, 13.5 |
| Miss 11.5 %, false 14 %, jitter 10 % | 0.65, 14.3 | 0.76, 14.2 |

Best pipeline on CV, all labels: r 0.686, prediction spread 66 % of label spread, residual log SD 0.138. Error by label decile: 16.6, 8.7, 7.3, 7.2, 7.8, 7.8, 8.1, 8.4, 9.9, 17.7 (label 0.29 to 0.50); the two outer deciles hold about a third of the error. By prediction decile, bias is flat at about -0.02.

Detector on the 25 sets (pre-only, trained on the external set): miss 0.150, false 0.165, jitter 0.215 of spacing, F1 0.862.

Sampling: about 150 cells give a standard error near 6.5 % on CV, 7.5 % on HEX, 3 % on CD if the cells were independent draws.

How far OOF can be trusted:

- Submission 1 test scores among random 1000-image draws of its OOF, grouped by donor: CD percentile 52, CV 61, HEX 94, mean 70.
- Old-format rows, natural experiment on fold assignment: rows with an adjacent ID in another fold against rows whose neighbours share their fold: CD -0.74 (CI -1.58, +0.05), CV +0.52 (CI -0.27, +1.27), HEX +0.39 (CI -0.42, +1.17). Adjacent IDs: label r 0.48, 0.41, 0.35; error r 0.38, 0.31, 0.22.
- Phase I (100 images) to Phase II (1000 images): V14 8.6206 to 8.6920, ec blend 8.7345 to 8.7816, LKSOM 2 8.7452 to 8.7805, ours 8.7010 to 8.9675 with a different model (crop package, then hybrid B, which OOF put 0.09 ahead of it). Whether the other three changed their models is unknown: an algorithm name can keep its title across new images. One model's mean has SD near 1.0 on 100 images.
- Full-data recipe A against 8 fold members on 60 external Topcon frames neither saw, member-level CI: CD -4.00 % (-9.71, +1.71), CV -0.43 % (-3.25, +2.39), HEX +0.53 % (-1.40, +2.45). Out of domain: fold members alone span CD -9.3 to +9.4 %. Cost of a scale shift on submission 1 OOF: HEX +0.39 needs about 3.4 % on the final output; the measured HEX shift at half weight, +0.27 %, costs +0.002. CD at half weight, -2 %, would cost +0.25, yet test CD did not rise above OOF. Labelled frames cannot repeat this: full-data members trained on all of them.

Readout noise: validation mean moves 0.059 between consecutive epochs of one run; within-run SD over epochs 11-20 is 0.052. Read the mean of the last five epochs, on at least two folds, paired. What was settled on weaker evidence: [assumption audit](ASSUMPTION_AUDIT_20261106.md).

## Where things are

- Packages: `submission_hybrid_20261030/` (submitted), `submission_hybrid_hex6_20261101/`, `submission_final_stretch_20261104/`. Records in `results/`.
- Pod `clear-ec-research-4gpu-1104`, `/workspace/clear-ec-research/experiments/meta_ceiling_20261103/`: `source/`, `sweep/` (graphs, best runs in `sweep/graph_sweep/` and `sweep/sim_pred/`).
- Deployed detectors: `experiments/green_track_20261028/deploy/detectors/` on the pod.

## Open decisions (PI)

- Start the package track: full-data graph fit, fp16 parity, container, T4 Try-out. As of 2026-11-11 18:00 EST the full-data fits and the fp16 parity are done; the container and the Try-out are not started. See "Open for the PI" in the freeze section: size and time limits (Q-001), docker build approval, use of the second submission.
- Whether to request the author-access endothelium datasets (Topcon SP-1P 1203 images, Rotterdam 383 images). Delivery time, cell labels and challenge rules are unverified.

## Ledger, after ideas are drafted

The PI deleted the local and pod archives on 2026-11-06. What remains, to be read only after ideas are drafted:

- Git commit `f8bc49b` on this branch: `docs/STATE_20261103.md`, `docs/IDEA_REGISTRY_20261031.md` and `results/meta_ceiling_20261103/REPORT.md` as of 2026-11-05, read with `git show f8bc49b:<path>`.
- Tests run after that commit exist only in the memory of session `local_5a5ca17b-018d-4ccd-b356-f8794b094404`: ask it whether a drafted idea repeats one.

A past failure rules out an idea only if the new idea repeats the same test on the same inputs.
