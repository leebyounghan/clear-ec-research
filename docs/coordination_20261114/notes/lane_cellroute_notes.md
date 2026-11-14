# Cell-route lane (device outline rule on detected cells), notes for sibling sessions

Pod folder `experiments/lane_cellroute_20261109/` (`RESULTS.tsv`, `REPORT.md`, `source/`, `feats/`, `out/`). CPU only.

## 2026-11-09 12:20 EST
- HEX: Konan k0/k2 statistics from a union point set (s46/s47/s48 pre-only TTA peaks >= 0.35 + Cellpose-SAM loose mask centroids,
  cross-source vote within 0.45 s, >= 1 source) as a nested ridge (alpha 100) on the log HEX residual of `lane_hex_20261108/preds/chain_cvw3.npz`:
  **HEX 9.7315 vs 9.7380 (-0.0065), folds -0.020/-0.008/-0.008/+0.007/-0.004, f04 -0.012.** Pre-registered from the 25-frame gate.
  Predictions `out/corr_hex_a100.npz` (P, BP row order). Features `feats/cellroute_feats.npz` (F, names; dataset index order, 8954 rows), columns `P2_*`.
- CV: closed for this route. Every point set is worse on the CV lane best (+0.019 to +0.029); raw r with the CV label only 0.2-0.34.
- Why detections cannot reach marker quality: detector peaks and mask centroids agree to 0.088 spacing; each is ~0.185 from the expert clicks.
  The device label is computed from noisy clicks (~0.16 s), and per-frame click scatter correlates with the label (CV +0.70, HEX -0.72, n=25).
- Running: click-noise emulation (mean over 8 draws of points + N(0, 0.16 s), the measured click scatter) on all frames, 3- and 5-detector unions.
  On the 25 frames it lifts HEX k2 r 0.679 -> 0.801 (3 det) / 0.828 (5 det; markers 0.845). ETA about 13:50 EST, then the HEX read.
- Mask centroids for all frames (x, y, area px): `feats/seg_centroids.npz` (indices, offsets, pts), from `lane_hex_20261108/seg_loose`.

## 2026-11-09 14:00 EST, wrap-up (cell-route lane)
- Kept: HEX 9.7315 vs 9.7380 (-0.0065) from the union-point Konan features. HEX lane input: `feats/HEX_input_P2_konan.npz` (idx, F, names; dataset index order).
  Kept predictions: `out/corr_hex_a100.npz`. Not robust across versions (others -0.0015 to +0.010).
- Closed:
  - CV: three versions worse (+0.015 to +0.029). Trimmed CV columns for a stack anyway: `feats/CV_input_trimmed_konan.npz`.
  - Click-noise emulation: the 25-frame HEX r 0.83 did not transfer (+0.007).
  - Box / patch regions for HEX: +0.006.
- Useful facts:
  - Gross-outlier trim (cell area within 0.4-2.5 × median) lifts raw r with the CV label from about 0.25 to 0.50 on all frames. Any area-CV feature without it is dominated by pseudo-cells.
  - The technician box (538×408) is predictable as the window with the most confident peaks (IoU median 0.68).
- No processes left running. REPORT.md and RESULTS.tsv (16 rows) are in the lane folder.

## 2026-11-09 17:35 EST (cell-route lane): teacher leak + deploy prep
- Teacher leak (for the backbone lane): with OOF teachers at --teacher-a 0.5, a student's fold-f read is optimistic by about 0.08-0.13 MAPE per metric.
  This was measured on a proxy learner at recipe A's memorization level (0.52) and is the same sign on all folds.
  The only clean option before the freeze is the fold-f checkpoints' in-sample predictions (option B; GPU inference).
  Tables and note: `experiments/lane_cellroute_20261109/teachers/` (LEAKAGE_NOTE.md).
- HEX step deploy prep: `FREEZE_CANDIDATE.md`. CPU parity checks pass. It needs sg2pre_s47/s48 trained on full data (about 10 GPU-min; asked the HEX lane for GPU 1).
  The ridge must be refit if the HEX lane's final chain differs from chain_cvw3.

## 2026-11-09 17:55 EST (cell-route lane)
- Label-free nested teacher for the backbone lane:
  - inputs: zero-shot Cellpose-SAM masks + image statistics, 59 columns;
  - plain OOF CD 7.448 / CV 10.718 / HEX 10.216, which beats the backbone's 46-column teacher on CD and CV;
  - files: `teachers/teacher_nested_LF_f{0,4}.npz`.
- Pre-only detector features (legible_window.npz, hex6) are NOT label-free. Each fold's detector was initialised from that fold's recipe A regression encoder.
- HEX step deploy: sg2pre_s47/s48 are trained. The 200-frame check shows features r 0.88-0.99 against OOF, and the corrected HEX is within 0.15 % of the OOF correction. FREEZE_CANDIDATE.md is updated.

## 2026-11-10 12:10 EST (cell-route lane): new-recipe inference entry
- Draft: `experiments/lane_cellroute_20261109/inference/` (recipe_infer.py = images -> CD/HEX, CV stub; members.py; params_build.py; tests).
- Fold-0 reproduction:
  - combine stage on all 1791 rows: exact (CD 0, HEX ~1e-15);
  - raw image end to end on 3 frames (CPU): CD 0.13 %, HEX 0.27 %.
- 14 missing pieces listed in REPORT.md "Missing pieces".
  - CV: apply fn, per-image graphs, frozen heads.
  - HEX: prior / ridge per image, g_hex46, cvw3.
  - CD: ViT-L, Phase-1 folder.
  - No owner: GRAPH / BEST / g_s10x3.
- Detector pair: everyone uses the green-track sg2pre_s47/s48. The cell-route duplicate was moved to deploy/archive_duplicate_pair_20261110/.

## 2026-11-10 13:25 EST (cell-route lane): inference entry v2
- All three columns come from the image. CV via cv_apply; HEX features via hex_test_features.build_arrays; ViT-L uses s46+s47; order-loss DINOv3 is in.
- Fold-0 parity:
  - combine stage: exact for every column;
  - end to end (CPU): CD 0.066 %, CV 0.066 %, HEX 0.35 / 0.48 %.
- Waiting on:
  - CV cv_image_parts parity;
  - HEX graph-net weights (about 14-15 EST) and cvw3 (about 3 h later);
  - 8 new detectors;
  - ViT-L full data.

## 2026-11-10 14:45 EST (cell-route lane)
- HEX step refit on chain_mtcnn5fix: HEX 9.7283 vs 9.7358 (-0.0075). New files: out/corr_hex_on_mtcnn5fix.npz and out/deploy_hexkonan_params.json.
- Inference entry v3:
  - CV per-image parts (GRAPH, quartets, ridge, frozen heads) are wired;
  - layer 1 is exact on the new CV and HEX candidates;
  - deploy smoke CV parts are within 0.004 log of the CV lane's in-sample deploy members.
- Open:
  - ViT-L full data (about 22:00 EST);
  - HEX graph nets / cvw3 / cps5;
  - 2 frozen-head detectors;
  - GPU run;
  - container.

## 2026-11-10 15:35 EST (cell-route lane; messaging paused, so read this note)
- To the CV lane: your pinning test passed. Stored fold-detector peaks -> my cv_parts vs cv_deploy_insample_members: G 2e-9, QP 1e-9, QO 2e-9, ridge 3e-6 / 9e-7. The entry is exact, and the deploy-smoke gaps are the detector swap.
- To the HEX lane: graph / best / g_s10x3 / g_hex46 / cvw3 / cps5 are wired per your FREEZE_CANDIDATE "Test-time path".
  - cps5 from the raw image with the fold-0 CPSAM run: 0.045 % / 0.073 %.
  - pre48_s10, full_hex_pre46_s5, cvw3 f99 and the CPSAM f99 member are picked up automatically once their files exist (manifest paths per your table).
- To the CD lane: ViT-L f99 paths are in the deploy manifest; the entry uses them as soon as the files exist.
- Cross-session messages from this session are paused until the user types here, so please leave anything for me in this folder.

## 2026-11-10 15:50 EST (cell-route lane), to the HEX lane: graph nets pinned
- Frame 32 stored fold-detector peaks -> my hex_features -> your gnn_hex_infer, against your fullfit_pred: pre47_s10 2.2e-9, pre48_s10 7.7e-10, hex46 2.2e-10.
- g_s10x3 / g_hex46 are live in deploy.
- cvw3 and the CPSAM f99 member are picked up automatically at the paths in your Test-time path table.
- Messaging from this session is still paused until the user types here.

## 2026-11-10 16:30 EST (cell-route lane), to the HEX and CV lanes and the coordinator: detector batch shape
- The stored training peaks (tta_infer_20261108, det_peaks_image, green track) were made with batch-4 loaders. bf16 heat maps depend on the batch shape.
  - det_peaks_image reproduces the stored peaks exactly at batch 4, but not at batch 1 (712 vs 714 peaks etc.).
  - A single test image (M=1, batch 1) therefore gets slightly different peaks. On GPU that moved prior_h6c 6-11 % and ridge_k3 up to 37 %, and HEX final up to 0.34 %.
- Fix in the inference entry (source/deploy_hexkonan_image.py):
  - the 4 TTA flips run as ONE batch of 4, and the identity-flip row is the single-view map;
  - the 13 other detectors run on 4 copies of the image.
  - Result: HEX features exact (difference 0) and HEX final ≤ 3e-5 vs stored OOF on GPU.
- HEX lane: hex_test_features / det_peaks_image on a 1-image memmap would hit the batch-1 path. My entry passes its own batch-4 peaks into build_arrays, so the entry is fine. Any other per-image use of det_peaks_image should pad to a batch of 4.
- 120-frame GPU fold-0 run in progress (ends about 19:20 EST). Container draft: experiments/lane_cellroute_20261109/inference/docker/ (Dockerfile, requirements, files.txt census 552 files / 31.1 GB, README).

## 2026-11-10 17:40 EST (cell-route lane), to the CD lane and the coordinator: GPU 3
- Read: s47 ViT-L first launch died of OOM at 17:01 EST on GPU 3. My fold-0 test (pid 387852, steady 13.5 GiB since 16:09) shares that GPU and may have contributed.
- Now: the test keeps running (steady memory, no new loads) and ends about 19:20 EST. I will start nothing else on GPU 3 until s47 is written (about 23:30 EST).
- The deploy-mode GPU run (20 frames, hex6 live, per-image time) moves to after s47, so both full-data ViT-L models will be real instead of stubbed. If the coordinator prefers it earlier on another GPU, say so here.
- cvw3 full data (HEX lane) is picked up at the manifest path, which is unchanged.

### 19:00 EST 11-10 — cell-route lane: full-data ViT-L s46 received (for the CD lane and the coordinator)
- CD lane's s46 (vitl16_f99_s46, raw): sha256 of epoch_08..11 and final.pt all match the list sent. Path matches my deploy manifest already, no change needed.
- Running now on the CPU pod (not GPU 3; 12 threads, pid 41885): strict load into my dinov3_l16t net, then my read vs the trainer's own stored predictions on its 2 held-out frames (predictions_epoch_08..11 single view, predictions_final_tta), then the s46 half of V on 8 fold-0 frames vs stored fold-0 V (sanity only). Result goes to out/vitl_f99_s46_smoke.json and here.
- GPU fold-0 parity run: 93/120 frames at 18:56 EST, ends about 19:55 EST. Still nothing new on GPU 3 until s47 is written.

### 19:05 EST 11-10 — cell-route lane: ViT-L s46 full data passes (for the CD lane and the coordinator)
- sha256 5/5 match, strict load OK, path already in my deploy manifest.
- My read against the trainer's stored predictions on its 2 held-out frames (CPU fp32 vs trainer GPU bf16):
  - single view e08-11 CD ≤ 0.48 %;
  - TTA4 final.pt CD ≤ 0.05 %;
  - the s46 half of V ≤ 0.18 %.
- On 8 fold-0 frames, the s46 half / stored fold-0 V has median 1.004 (0.989-1.072). Different weights, so this is a sanity check only.
- Waiting for s47. Then V goes live, followed by the 20-frame deploy GPU run on GPU 3 (after s47 frees it).

### 19:55 EST 11-10 — cell-route lane: GPU fold-0 reproduction, 120 frames (for the coordinator)
- The entry reproduces stored fold-0 OOF on GPU over 120 fold-0 validation frames, using fold-0 models. Maximum relative difference:
  - CD 0.100 % (one frame at 0.1 %, p95 0.012 %);
  - CV 0.047 %;
  - HEX final 0.014 %.
- Neural members stay within 0.18 % (bf16 noise). P1, mask and every HEX/cell-route feature is within 4e-7.
- Stubs: the 15 full-data-only inputs (f6/have, CV parts, graph/best, g_s10x3/g_hex46, dino_cvw3). The deploy smoke and pinning tests cover them.
- Median 127 s per frame on a GPU shared with a ViT-L training run. File: out/gpu_fold0_120.json.
- My GPU-3 test is finished; GPU 3 now only holds the CD lane's s47. The next GPU work is the 20-frame deploy run after s47 is written.

### 20:05 EST 11-10 — cell-route lane: s47 now ~00:15 EST (for the coordinator)
- The CD lane says full-data ViT-L s47 lands about 00:15 EST 11-11, not 23:30. My GPU-3 test ended at 19:55, so s47 now has GPU 3 alone.
- My watcher picks up s47's exit.json, checked every 10 min. The 20-frame deploy GPU run starts after that, at about 00:20-00:30 EST, and ends about 01:15 EST. CPSAM f99 stays stubbed if it has not landed by then; I rerun those frames when it does.

### 21:45 EST 11-10 — cell-route lane: s47 received, deploy GPU run started (for the CD lane and the coordinator)
- CD lane's full-data ViT-L s47 was written at 21:37 EST. s47 is now checked on the CPU pod the same way as s46: sha256, strict load, the trainer's stored held-out predictions.
- The 20-frame deploy GPU run started at 21:44 EST on GPU 3 (pid 422128, 4 CPU threads). Setup:
  - full-data weights;
  - both ViT-L seeds live (no V stub);
  - hex6 live;
  - cps5 stubbed, because CPSAM f99 only has epoch_08 so far.
  - GPU 3 is shared with the backbone lane's dinov3_b16 (21 GB) and a keepalive.
- Output: out/gpu_deploy_20.json, which reports the deploy vs OOF gap per column, hex6 f6 against the stored OOF f6, and seconds per frame. Expected end is about 22:30-23:00 EST.

### 21:55 EST 11-10 — cell-route lane: ViT-L s47 passes; full V matches the trainer's own predictions (for the CD lane)
- s47: sha256 5/5 match and strict load works. Checked against the trainer's stored held-out predictions (2 frames, my CPU fp32 vs trainer GPU bf16):
  - single view ≤ 1.05 % (one read; the rest ≤ 0.35 %);
  - TTA4 ≤ 0.19 %.
- Full V (the geometric mean of the halves) against the same rule on stored predictions: +0.12 % / −0.15 %. Nothing off on the CD side. CD has no stub left.
- On 8 fold-0 frames, full V / stored fold-0 V has median 1.008 (0.978-1.078). Different weights, so this is a sanity check only.

### 22:35 EST 11-10 — cell-route lane: deploy GPU run done; container is 48 GB (for the coordinator)
- **Deploy run** (full-data weights, 20 training frames, 4 per fold; hex6 and both ViT-L seeds live; only cps5 stubbed): runs end to end on GPU.
  - In-sample against stored OOF (not parity): median gap CD 0.44 %, CV 1.27 %, HEX 1.30 %; maximum CD 2.6 %, CV 6.0 %, HEX 4.8 %.
  - hex6 f6 against OOF f6: correlation 0.952, and the `have` flag agrees on 20 of 20 frames.
  - Median 74 s per frame on a shared GPU 3.
- **Container census on GPU:** 769 files, 48.3 GB of fp32 weights. The first submission was 4.9 GB.
  - Biggest folders: ViT-L 13.9 GB, recipe A 8.9 GB, P1 7.3 GB, HEX deploy 5.6 GB, CV arms 5.2 GB, detectors 4.1 GB.
  - **Decision needed:** someone must check the Grand Challenge image and model-tarball size limits before any build.
  - Possible shrink, not done and needing a parity test: bf16 storage of linear and conv weights, which should be exact under GPU autocast.
- **Open:** CPSAM f99 (epoch_10-12 not on disk yet; 08 and 09 are). When it lands, I rerun the 20 deploy frames without the cps5 stub.
- **Infra note:** `kubectl cp` to clear-ec-cpu-1109 failed with "error: EOF" at about 22:25 EST while exec still worked. My push script now falls back to exec-stdin and checks md5.

### 23:20 EST 11-10 — cell-route lane: size / runtime table ready (coordinator task 1, 3, 4); fp16/bf16 test running (task 2)
- File: experiments/lane_cellroute_20261109/inference/docker/size_runtime_budget.md (+ size_groups.json, dup_check.json, budget_analysis.json).
- Totals: 769 files, 51.9 GB fp32 (CPSAM f99 epochs 10-12 counted at epoch 08 size). First frame 159 s, steady 81 s on shared GPU 3, peak GPU 17.0 GB.
- Biggest groups:
  - ViT-L 12.1 GB (CD);
  - recipe A e8-19 8.2 GB (CD only, A_cd half);
  - P1 7.3 GB (all 3 columns);
  - CPSAM 512 cps5 6.1 GB (HEX);
  - detectors 5.5 GB unique;
  - DINOv3-B CV 3 seeds 5.2 GB (CV).
- Where the time goes: weight loading (state dict to GPU) is 29.6 s of the steady 81 s, versus 7.7 s of member forwards. HEX features take 20.9 s (11.5 s CPU feature build + 9.2 s for 125 graph nets). A GPU alone would save only about 4-8 s (estimate). GC runs one image per job, so each job costs about the first frame, 159 s.
- No-OOF-cost savings:
  - the duplicate CPSAM file (seg = P1 backbone, same sha) 1.2 GB;
  - base weights read only to build architectures 1.9 GB (needs a one-frame check);
  - half-precision storage (test running, results by about 00:10 EST).
- Snapshot rules (task 3): the lanes average predictions (log means over epochs, flips and seeds), not weights.
  - ViT-L has weight-averaged swa6 / ema tracks already trained for every fold and for full data, with predictions_final_tta stored. The CD lane can price "1 SWA file per seed" (−9.7 GB) from stored predictions, with no training.
  - The GB saved per cheaper rule is in section 3 of the file.
- fp32 is deterministic run to run (CD identical on 20/20 frames vs the earlier deploy run, CV ≤ 8e-9), so the fp16 / bf16 differences will be pure storage effects.

### 23:45 EST 11-10 — cell-route lane: base weights not needed (−1.9 GB, verified)
- With every tensor read from the base files replaced by noise (timm ViT-L, DINOv3-B, CPDINO ViT-B; 1.90 GB), outputs are bit-identical. This holds for recipe A, DINOv3-B and ViT-L members and for a full detector's heat map.
- So the strict loads overwrite them fully, and only the configs are needed. Table updated in docker/size_runtime_budget.md.
- No-cost savings so far: 1.9 GB base weights + 1.2 GB duplicate cpsam_v2 = 3.1 GB. Half precision is still running.

### 23:50 EST 11-10 — cell-route lane → backbone lane: no need to hold the CV build
- Thanks for the hold-file note. My timing numbers come from the fp32 run (03:37-04:06 UTC), which is already done.
- The fp16 and bf16mix runs only compare values against fp32, and time-slicing does not change values. So let the CV lane's DINOv3 s47 build start on GPU 3 as planned. The bf16mix run will launch through your guard after fp16 (about 05:00 UTC) and needs about 17 GB.

### 00:40 EST 11-11 — cell-route lane: half precision for everything fails CV and HEX (task 2, interim)
- 20 deploy frames, change against fp32, median / max per column:
  - fp16: CD 0.013 / 0.032 %, CV 0.168 / 0.49 %, HEX 0.381 / 1.83 %.
  - bf16mix: CD 0.006 / 0.045 %, CV 0.158 / 0.84 %, HEX 0.541 / 2.16 %.
- Under your rule (median < 0.05 %), CD passes; CV and HEX fail. So half precision for everything is not the default.
- Cause:
  - The regression members barely move (each member's own output: median 0.003-0.05 %, max ≤ 0.27 %).
  - The thresholded parts move: detector peak counts change by up to 15 per frame, the hex6 feature f6 by a median of 1.8 %, and the graph-net outputs on the peaks change too. The segmentation mask is thresholded the same way.
- Running now on GPU 3 through the backbone guard (`run_budget2_20261111.sh`), about 1 h:
  - fp16-nodet: half for all regression members (ViT-L, recipe A e8-19, P1 seeds, cps5, D, DINOv3-B CV, order-DINOv3, cvw3); detectors, hex6, frozen heads with their recipe A e20 encoder, `cpsam_v2` and the graph models stay fp32. Saves 20.6 GB if it passes.
  - fp16-cdonly: half for ViT-L and recipe A e8-19 only. Saves 10.2 GB.
  - After both: the no-stub deploy rerun (`test_gpu_deploy.py 20 gpu_deploy_20b`), once the CPSAM f99 `complete.json` exists. The chain waits up to 2 h for it.
- File: `docker/size_runtime_budget.md`, sections 2a and 2b.
- Housekeeping: a push helper mistake left a stray copy of a small script at `lane_cellroute_20261109/experiments/...`. I moved it to `lane_cellroute_20261109/archive/stray_push_20261111/`. Nothing was deleted.

### 01:55 EST 11-11 — cell-route lane: half precision for the regression members passes (task 2 done); deploy about 28 GB
- fp16-nodet: all regression members are stored in fp16 (ViT-L, recipe A e8-19, P1 seeds, cps5, D, DINOv3-B CV, order-DINOv3, cvw3). The detectors, hex6, frozen heads with their recipe A e20 encoder, `cpsam_v2` and the graph models stay fp32.
  - 20 deploy frames against fp32, median / max: CD 0.008 / 0.029 %, CV 0.005 / 0.024 %, HEX 0.009 / 0.052 %.
  - That passes your rule (median < 0.05 %) on all three columns, so it is the default. It saves 20.6 GB.
  - Peak counts, f6, the cell-route features and the graph features are bit-identical to fp32. The detectors alone caused the failure of fp16 for everything.
- fp16-cdonly (ViT-L and recipe A e8-19 only): CD 0.007 / 0.023 %; CV and HEX unchanged at noise level. Saves 10.2 GB, a fallback if you want the smallest change.
- Size with no OOF cost: 51.94 − 1.90 base weights − 1.23 duplicate `cpsam_v2` − 20.57 fp16 ≈ **28.2 GB**.
  - Of that, 7.7 GB stays fp32: detectors and hex6 5.5, `cpsam_v2` 1.2, recipe A e20 0.7, graph models 0.3.
  - **For the 09:00 questions:** every GB-saved figure in section 3 halves under this default (for example, dropping ViT-L e8-11 single-view saves 4.85 GB, not 9.7).
- Deploy form: a one-off conversion writes half copies, and the entry calls `.float()` at load.
  - Not done. It writes about 20 GB of new files, which I won't do without your OK. It is also a packaging step, and the build is not mine to start.
- CPSAM 512 full-data, epochs 08-12 (HEX lane message received): all load strictly. My CPU single-view output matches the stored predictions within 0.38 % for HEX.
  - The no-stub deploy rerun started on GPU 3 at 06:38 UTC (01:38 EST); results by about 02:15 EST.
- File: `docker/size_runtime_budget.md`, sections 2a-2c and Conclusion.

### 02:20 EST 11-11 — cell-route lane: deploy run with no stubs (to the coordinator and HEX lane)
- HEX lane: thanks, cps5 full-data received. The deploy manifest already had the epoch_08..12 paths, and my call is the same as yours (mean of the log predictions, then exp).
  - All 5 epochs load strictly. My single-view read matches your stored predictions within 0.38 % for HEX.
- Deploy GPU run, 20 frames, every member live (`out/gpu_deploy_20b.json`). Deploy against stored OOF, median / max |rel|:
  - CD 0.44 / 2.6 %, identical to the earlier run.
  - CV 1.27 / 6.0 %, identical to the earlier run.
  - HEX 1.67 / 7.2 %, was 1.30 / 4.8 % with cps5 stubbed.
- The HEX change is cps5 full-data against cps5 OOF on these frames: r 0.878, mean log ratio −0.0068. The HEX lane's 64-frame check gave r 0.954 and −0.0072. These are training frames for the full-data model, so this is not a parity number.
- Time 92 s median and 165 s for the first frame, on GPU 3 shared with the CV build.
- My GPU 3 chain is finished; I have nothing running on GPU 3 now.

### 03:15 EST 11-11 — cell-route lane → CV lane, coordinator: CV 6-component freeze is in the entry; fold-0 parity passes
- CV lane, thanks. Your new `deploy/` params (md5 3b437500) are loaded. My entry reads that folder live, and its recipe A and Phase-1 CV scalars equal yours (0.989108 / 0.987972).
- Entry change:
  - Plain DINOv3 (dino_logcv, s46/s47/s49 e16-20) and order-DINOv3 run only when the CV weights use them, so order-DINOv3 is no longer opened.
  - The CV assembly is one method (`Recipe.cv_column`).
- `params/fold0.json` rebuilt from the new `cv_inputs_oof.npz` (w0, sA[0], stretch0). No other params or manifests file changed. The old copy is in `archive/params_before_cv6_20261111/`.
- Fold-0 parity:
  - Formula, all 1791 fold-0 rows: equal to `apply_fold` exactly, and the stored `cv_oof` within 5.6e-16.
  - Live, 120 frames on GPU 3, with A, P1 and plain DINOv3 run on fold-0 weights: CV median 0.011 %, max 0.047 %. A_cv max 0.13 %, dino_logcv max 0.074 %, P1 ~1e-7. Same as the previous stack.
- Size: order-DINOv3 dropped saves 0.86 GB at fp16, so the deploy is now about 27.3 GB (`docker/size_runtime_budget.md` 2c).

### 04:40 EST 11-11 — cell-route lane → CD lane, coordinator: two-seed D is in the entry; fold-0 CD parity passes
- CD lane, thanks.
  - `members.dino_cd` now takes per-seed lists: D = exp(mean over seeds of the mean log over 4 flips × e16-20).
  - Fold 0 uses s46 (lane_gpu3) + s47 (lane_gpu2), the runs your `cd_read.roots` picks, with `freeze_cd_params_20261111.json` per_fold 0.
- Formula, all 1791 fold-0 rows: my `cd_column` equals your `oof_cd` exactly; all 5 folds are exact with their params. The stored D equals geomean(D46, D47), and the five-fold CD is 6.1741.
- Live, 120 fold-0 frames on GPU 3: D median 0.002 %, max 0.087 % (s47 alone max 0.20 %, bf16 noise as for A and V). CD column median 0.0002 %, max 0.008 %.
- Deploy:
  - Stays on s46b + the 20261109 params until your full-data s47 `complete.json` exists. `params_build.py` then switches to s46b + s47 + the 20261111 deploy params by itself.
  - When it lands, I'll run sha256 (to compare with your list), a strict load, and the single-view check against your stored predictions, then rebuild.
  - If it is not built by 15:00 EST, the fallback is already in place (the coordinator's rule).

### 05:40 EST 11-11 — cell-route lane → CD lane, coordinator: deploy D is the two-seed mean with freeze_cd_params_20261111
- CD lane, thanks. All five sha256 match your list, and every epoch loads strictly.
- My deploy D on GPU (`members.dino_cd`, production path) against your reference:
  - 2019-0064-ODCN: 1082.50 vs 1082.03 (+0.04 %).
  - 2019-0064-OSCN: 1517.17 vs 1518.17 (−0.07 %).
  - Single seeds: s46b +0.29 / −0.04 %, s47 −0.20 / −0.09 %. That is bf16 noise, the size of my 120-frame fold-0 parity (s47 max 0.20 %).
- `params_build.py` rerun. Deploy has `dino_cd` = s46b + s47 epochs 16-20, and `cd` equals your `freeze_cd_params_20261111.json` deploy block exactly. Only deploy params and manifests changed; the old copy is in `archive/params_before_d2deploy_20261111/`.
- Coordinator: the deploy is now about 28.2 GB at the half-precision default (+0.86 GB for s47; order-DINOv3's −0.86 GB from the CV freeze cancels it). The deploy GPU run has not been redone for the CD and CV changes; tell me if you want a 20-frame no-stub rerun before the freeze.
- Correction to the line above: I started the 20-frame no-stub deploy rerun myself (GPU 3, guard, launched 10:36 UTC, `out/gpu_deploy_20c.json`), so there is nothing to decide. Results by about 06:15 EST.

### 06:15 EST 11-11 — cell-route lane: no-stub deploy run with the current CV and CD stacks
- 20 training frames, every member live (CV 6-component stack, two-seed D, `freeze_cd_params_20261111`), `out/gpu_deploy_20c.json`.
- Deploy against stored OOF, median / max |rel|:
  - CD 0.51 / 2.4 % (was 0.44 / 2.6 % with the old stacks).
  - CV 1.32 / 5.7 % (was 1.27 / 6.0 %).
  - HEX 1.67 / 7.2 % (unchanged).
- 92.5 s median per frame on shared GPU 3. These are training frames for the full-data models, so this is not a parity number; the fold-0 parities for both changes passed earlier.
- My GPU 3 work is finished; nothing of mine is running.

### 06:35 EST 11-11 — HEX lane → cell-route lane: new HEX chain file, please refit (coordinator decision after the seed audit)
- New chain: `experiments/lane_hex_20261108/preds/chain_seedaudit10.npz` (c0..c4, final_hex, idx, fold, steps). Steps: ap1 h6w graphens imgstats prior ridgeev ridgeev_fall16 segm cvw3m_fix mtcnn5_fix. HEX 9.73258 five folds / 9.86437 folds 0+4. It replaces `chain_mtcnn5fix.npz` (9.7358), on which the 9.7283 refit was fitted.
- Changes: `gnn:hex_pre46_s5` removed (g_hex46 input no longer used); cvw3 read as the mean log over seeds 46 and 47. Test time: dino_cvw3 = exp(family_infer.predict of the 10 checkpoints `deploy/cvw3_f99_s46` and `deploy/cvw3_f99_s47`, `runs/fold99/dinov3_b16/epoch_{08..12}.pt`)[:, 2]); the s47 f99 build lands about 07:00 EST.
- Deploy params: `preds/deploy_params.json` (stretch c 0.99012; the 11-step version is in `archive/`). `source/hex_apply.py oof` reproduces the chain to 8.9e-16.
- Please send the refit value to the coordinator and the HEX lane (also sent by message).

### 06:30 EST 11-11 — cell-route lane → coordinator and HEX lane: cell-route refit on chain_seedaudit10 = HEX 9.7262
- **Refit value (five folds, nested, recipe unchanged): HEX 9.72623**, against chain 9.73258 (−0.00635). Folds 0+4: **9.8554** against 9.86437 (−0.0090). Per fold −0.017 / −0.008 / −0.008 / +0.001 / −0.001; 8845 rows corrected. File: `out/corr_hex_on_seedaudit10.npz`. It replaces 9.7283 on `chain_mtcnn5fix`.
- HEX lane: the entry reads your new `deploy_params.json` (10 steps, stretch c 0.99012).
  - hex46 is no longer run, because the entry runs it only if the chain has `gnn:hex_pre46_s5`.
  - Deploy `dino_cvw3` = family mean over the 10 checkpoints `cvw3_f99_{s46,s47}` e08-12, HEX column, as you specified.
  - Fold-0 tests keep `dino_cvw3` stored, because the fold-0 s46 cvw3 run kept no e08-12 checkpoints.
- Layer 1, all 1791 fold-0 rows, all current stacks: CD 0, HEX chain 1.3e-15, HEX cell-route 1.2e-15, CV 5.6e-16.
- When `cvw3_f99_s47` lands (about 07:00 EST), a waiter runs: sha256, strict load and single-view check, the deploy `dino_cvw3` on your held frames, then the 20-frame no-stub deploy run (`out/gpu_deploy_20d.json`).

### 07:35 EST 11-11 — cell-route lane → HEX lane, coordinator: cvw3 s47 checked; deploy runs with every 11-11 stack
- cvw3 s47 full data: e08-12 load strictly; single view vs your stored predictions within 0.22 % (HEX).
  - sha256 prefixes: 79022f60, 0e92f486, 10633cad, 978e5524, f9af399a.
  - Deploy `dino_cvw3` on the held frames 4569 / 4570: 0.4101 / 0.4245 (s46 0.4180 / 0.4319, s47 0.4023 / 0.4171). HEX lane: compare with your reference if you have one.
- No-stub deploy run, 20 frames (`out/gpu_deploy_20d.json`), with the CD two-seed D, the CV 6-component stack, and HEX seedaudit10 plus the refit cell-route step:
  - Deploy against OOF, median / max |rel|: CD 0.51 / 2.4 %, CV 1.32 / 5.7 %, HEX 1.58 / 7.0 % (was 1.67 / 7.2 % on the old chain).
  - 92.4 s median per frame.
- Current freeze values from my side: HEX with the cell-route step 9.72623 (five folds), 9.8554 (folds 0+4).
- Nothing of mine is running on GPU 3. Still open for the coordinator: the OK to write the half-precision copies (about 20 GB of new files) and packaging.

### 09:55 EST 11-11 — cell-route lane → coordinator: parity passes on the current stacks; fp16 copies written (21.4 GB)
- Re-measure on today's stacks, fp16-nodet against fp32 on 20 deploy frames, median / max: CD 0.009 / 0.027 %, CV 0.005 / 0.021 %, HEX 0.006 / 0.030 %. That passes your rule.
- Copies: 77 files, 42.85 GB → 21.43 GB, in `lane_cellroute_20261109/deploy_fp16/`. Every one is bit-identical to the tested simulated cast; nothing was overwritten.
  - The detectors, hex6, `cpsam_v2`, recipe A e20 and the graph models stay fp32 and were not copied.
- Runtime on the current stacks: steady 90.0 s (HEX graph nets 17.1 s for 100 nets, up from 9.2 s; unexplained, contention suspected), first frame 213 s.
- Packaging is not started: pointing the entry at the copies, skipping the base weights, and docker all wait for the PI.

### 10:20 EST 11-11 — HEX lane → cell-route lane: HEX freeze is now the joint nested ridge, please refit (coordinator)
- New OOF file: `experiments/lane_hex_20261108/preds/chain_joint.npz` (c0..c4, final_hex, idx, fold): HEX 9.72621 five folds / 9.85123 folds 0+4 (strictly nested 9.72662).
- Deploy params `preds/deploy_params_joint.json`; entry `source/hex_apply.py` `hex_any(P, I, f=None)` (f=None deploy, f=k nested fold k; nested parity 2.2e-16); math in `source/joint_deploy.py` `apply_joint`. Inputs as `hex_apply.oof_inputs` (g_hex46 is used again). Deploy stretch g 1.025, c 0.99411, mu -0.61800.
- Fallback stays: `chain_seedaudit10.npz` + `deploy_params.json`. A 39-column version with the backbone attention-pool member may follow if its full-data build is verified by 16:30 EST.

### 10:30 EST 11-11 — cell-route lane → coordinator, HEX lane: refit on chain_joint; strictness; entry on hex_any
- Kept-recipe refit: **9.71154 (−0.0147)**, folds 0+4 9.84609 (−0.0051). Per fold −0.023 / −0.028 / −0.026 / −0.009 / +0.013. File `out/corr_hex_on_joint.npz`.
- Strictly nested (version choice among 15 versions + "no step" by inner CV on training folds): **9.72169 (−0.0045)**, folds 0+4 9.85786 (**+0.0066**).
  - Without the 25 gate frames: kept −0.0094, strict +0.0003.
  - Non-strict parts: the version was chosen on all-label reads, and the point sets were pre-registered from the 25-frame gate.
  - Script `source/corr_strict.py`. Sent to the coordinator in full; the decision is theirs.
- HEX lane: my entry calls your `hex_apply.hex_any(P, I, f)` with `deploy_params_joint.json` embedded (fold 0 = nested[0], deploy f=None). g_hex46 and hex46 are back. Layer 1 on fold 0 matches `chain_joint` to 4.4e-16.

### 10:55 EST 11-11 — HEX lane → cell-route lane: joint ridge confirmed (coordinator, strict numbers); deploy params rebuilt
- The OOF file is unchanged: `preds/chain_joint.npz` (9.72621 nested, 9.72662 strict) — refit on it as in the 10:20 note.
- `preds/deploy_params_joint.json` was rebuilt so the deploy fit is the nested procedure with no outer fold (base on the fit rows = all-rows BEST in-sample from the CV lane's `cv_deploy_insample_members.npz`): deploy alpha 1, stretch c 0.99584, g 1.025, mu -0.61800. Entry unchanged: `hex_apply.hex_any(P, I, f=None)`; nested parity 3.3e-16. The earlier params (alpha 100, c 0.99411) are archived.

### 11:40 EST 11-11 — HEX lane → cell-route lane: HEX freeze is now the 39-column joint ridge (attention-pool column added)
- New OOF file to refit on: `experiments/lane_hex_20261108/preds/chain_joint_attnp.npz` (c0..c4, final_hex, idx, fold): HEX 9.71882 five folds / 9.84826 folds 0+4 (strict 9.71977). It replaces `chain_joint.npz` (38 columns, fallback 1).
- Params `preds/deploy_params_joint_attnp.json` (deploy = the nested procedure with no outer fold: alpha 1, stretch c 0.99561, g 1.025, mu -0.61800). Entry `hex_apply.hex_any(P, I, f=None)`; extra input `I['attnp5']` = exp of the mean log HEX over the backbone lane's attention-pool f99 raw epochs 8-12 (call, paths and sha256 in `backbone_attnpool_f99_handoff.md`). Nested parity 2.2e-16.
