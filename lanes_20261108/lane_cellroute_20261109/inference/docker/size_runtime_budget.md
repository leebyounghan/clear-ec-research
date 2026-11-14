# Deploy size and runtime budget (cell-route lane, 2026-11-11)

All numbers are measured, except where a line says "estimate". Sources:
- `docker/size_groups.py` → `size_groups.json`: os.stat of the deploy census, no du.
- `docker/dup_check.py` → `dup_check.json`.
- `test_budget.py` → `out/budget_fp32.json`: 20 deploy frames on GPU 3, full-data weights, nothing stubbed. Stage timers are CUDA-synchronised.
- `analyze_budget.py` → `budget_analysis.json`.
- cps5 used the fold-0 CPSAM 512 weights: same architecture and file size, because f99 was not complete at launch.
  - f99 finished at 05:55 UTC. All five epochs are 1,218,437,542 bytes, the size counted in the census.
  - The no-stub deploy run (`out/gpu_deploy_20b.json`) took 92 s median and 165 s for the first frame on shared GPU 3.

## Conclusion

- **Size:** the deploy reads 769 files, 51.9 GB, all fp32 weights. This counts CPSAM f99 epochs 10-12 at the size of epoch 08.
- **Biggest groups:**
  - ViT-L: 12.1 GB, CD only.
  - recipe A epochs 8-19: 8.2 GB, CD only.
  - P1: 7.3 GB.
  - CPSAM 512 (cps5): 6.1 GB, HEX only.
  - The detectors together: 5.5 GB unique.
- **Runtime per frame on GPU 3** (shared with a training job):
  - The first frame takes 159 s. In Grand Challenge every job is one image, so each job pays roughly this first-frame cost.
  - Later frames take 81 s (median of 19).
- **Peak GPU memory:** 17.0 GB.
- **Where the time goes:**
  - Moving weights into the networks costs 29.6 s per frame: state-dict copy plus host-to-GPU transfer.
  - Neural forward passes of the members cost 7.7 s.
  - HEX features cost 20.9 s: CPU feature build 11.5 s, plus 125 small HEX graph nets 9.2 s.
  - Size, not GPU compute, drives the runtime.
- **Savings with no change to the recipe:**
  - The CPSAM backbone file is stored twice, as the segmentation model and as the P1 backbone. Removing the duplicate saves 1.2 GB.
  - Base weights read only to construct architectures save 1.9 GB. Verified: with every base tensor replaced by noise, outputs are bit-identical (`test_base_weights.py`, details in section 1).
  - **Half-precision storage of the regression members (fp16-nodet) passes the 0.05 % median rule on every column and saves 20.6 GB** (section 2b). Median / max change: CD 0.008 / 0.029 %, CV 0.005 / 0.024 %, HEX 0.009 / 0.052 %.
    - The detectors, hex6, `cpsam_v2`, recipe A e20 and the graph models stay fp32.
    - Half precision for everything fails CV (0.17 %) and HEX (0.38 %), because thresholded detector peaks flip (section 2a).
  - **Total after these three savings: about 28.2 GB** (section 2c). Every GB figure in section 3 halves under this default.
  - **The CV lane's 6-component freeze (11-11) dropped order-DINOv3:** another 0.86 GB at half precision and 1.9 s per frame.
  - **The CD lane's seed audit added D's second seed (s47):** +0.86 GB and about 2.2 s.
  - **Total with both: ≈ 28.2 GB.**
- **Savings that need the lanes' OOF price** (section 3):
  - ViT-L epoch 8-11 single-view files: 9.7 GB. An SWA weight track per seed already exists for every fold, so its OOF is readable from stored predictions.
  - recipe A epochs 8-19 for A_cd: 8.2 GB.
  - CPSAM 512 epochs: 4.9 GB.
  - DINOv3-B CV extra seeds: 3.4 GB.
  - 1 file instead of 5 epochs for each of D, order-DINOv3 and cvw3: about 1.4 GB each.

## 1. Table per member group

- **Time** is seconds per frame:
  - "steady" is the median of frames 2-20, with weights already in RAM but copied into the network and moved to the GPU on every frame;
  - "first" is frame 1, including disk reads and model builds.
- **load / fwd:** for regression members, the steady time splits into state-dict copy plus host-to-GPU transfer (load) and forward passes (fwd).
- **GPU 3 was shared** with a training job at 100 % utilisation throughout. The forward passes add up to only 7.7 s, so an unshared GPU would save a few seconds at most (estimate).

| member group | files | GB (fp32) | steady s (load / fwd) | first s | feeds |
|---|---|---|---|---|---|
| ViT-L s46+s47 (epochs 8-11 single view + final TTA4, per seed) | 10 | 12.13 | 8.4 (7.6 / 0.9) | 16.4 | CD |
| recipe A s46+s47 epochs 8-19 (single view, A_cd half) | 24 | 8.24 | 11.3 (8.2 / 3.1) | 15.3 | CD |
| recipe A s46+s47 epoch 20 (TTA4) | 2 | 0.69 | 1.7 (0.7 / 1.0) + tokens 0.2 | 4.3 | CV, HEX (A); CD (A_cd half); CV frozen heads; encoder of the 4 frozen detector heads |
| P1 five members + CPSAM backbone | 6 | 7.33 | 9.6 | 16.5 | CD, CV (h_cv), HEX (P1, and the spacing for the HEX prior) |
| CPSAM 512 cps5 epochs 8-12 | 5 | 6.09 | 4.8 (3.1 / 1.6) | 11.2 | HEX |
| DINOv3-B s46b epochs 16-20, TTA4 (D; also CV seed 1) | 5 | 1.72 | 2.2 (1.7 / 0.5) | 6.8 | CD; CV |
| DINOv3-B CV, 2 more seeds, epochs 16-20 (single view, 3 seeds read) | 10 | 3.43 | 5.3 (4.9 / 0.4) | 6.7 | CV |
| order-DINOv3-B epochs 16-20 | 5 | 1.72 | 1.9 (1.8 / 0.1) | 2.6 | CV |
| cvw3 DINOv3-B epochs 8-12 | 5 | 1.72 | 1.7 (1.6 / 0.1) | 2.5 | HEX |
| CPSAM segmentation (`cpsam_v2`) + seg features | 1 | 1.23 (same bytes as the P1 backbone) | 1.7 (features 0.5, CPU) | 5.1 | CD (mask), HEX (seg features, cell-route points) |
| pre-only detectors s46/47/48 (TTA4 as one batch of 4) | 3 | 1.03 | 1.9 | 6.2 | HEX (cell-route, prior, ridge, graph-net graphs), CV (GRAPH, quartets, sim ridge) |
| 9 full + 4 frozen-head detectors (all16, single view) | 13 | 3.09 (heads 0.004) | 6.9 | 23.4 | HEX (all16 features) |
| hex6 corrector (green-track deploy, 12 detectors; 8 shared with the rows above) | 14 | 4.12 (1.37 not shared) | 1.5 | 13.9 | HEX (f6, have) |
| HEX features (CPU, workers=1) + 125 HEX graph nets | 125 | 0.06 | 20.9 (features 11.5, nets 9.2) | 23.9 | HEX (g_s10x3, g_hex46, prior/ridge features) |
| CV graph nets (200) + sim nets (3) + CV frozen heads (60) | 263 | 0.20 | 1.2 | 3.7 | CV (graph, quartets, ridge, fz_*); HEX (graph, best, g_s10x3) |
| cell-route features (Konan cells on the detector points) | 0 | 0 | 0.3 | 0.4 | HEX |
| base weights read only to construct architectures (timm ViT-L, DINOv3-B, CPDINO ViT-B) | 4 | 1.90 | in "first" | – | none: fully overwritten (verified, see below) |
| code, params, small files | ~280 | 0.01 | – | – | – |
| **total (unique files)** | **769** | **51.94** | **80.8 median** (sum of stages 81.3) | **159.0** | |

Notes:
- **Unique detector bytes:** pre-only 1.03 GB + 9 full 3.09 GB + hex6-only 1.37 GB = 5.5 GB.
- **The base weights** are opened in these build stages:
  - `V/load` (timm ViT-L, 1.21 GB);
  - `D/load` (DINOv3-B, 0.34 GB);
  - every CPDINO ViT-B build: recipe A, tokens, 16 detectors and hex6 (0.34 GB).
  - In each case a strict `load_state_dict` of the full network follows. **Verified 2026-11-11** (`test_base_weights.py`, CPU, frame 1286): every float tensor read from the base files was replaced by random noise.
    - recipe A, DINOv3-B and ViT-L outputs are bit-identical (191, 211 and 318 base tensors randomised).
    - One full detector's heat map is bit-identical (191 tensors randomised).
    - So only the architecture is needed: config files of a few kB. The entry can skip reading these weights with the same hook (return zeros, or build with pretrained off), saving 1.9 GB and the base-file reads in every CPDINO build (16 detectors + hex6 + recipe A).
- **Disk reads:** frame 1 read 51.6 GB of weights in 29.0 s (1.8 GB/s, from the pod's page cache). A cold disk in a fresh container would be slower; I did not measure that.
- **Deterministic:** the fp32 run reproduces the earlier deploy run. CD matches exactly on 20 of 20 frames, CV within 8e-9.

### 1b. Changes from the 11-11 freezes (sizes measured by os.stat; seconds are estimates until the re-measure, section 4b)

| member group | change | files | GB (fp32) | steady s per frame | feeds |
|---|---|---|---|---|---|
| DINOv3-B D (CD lane seed audit) | + s47 e16-20 TTA4, now s46b + s47 | 5 → 10 | 1.72 → 3.43 | 2.2 → ~4.4 | CD (s46b also CV) |
| cvw3 (HEX lane seed audit) | + s47 e08-12, now s46 + s47 | 5 → 10 | 1.72 → 3.43 | 1.7 → ~3.4 | HEX |
| order-DINOv3 (CV 6-component freeze) | removed | 5 → 0 | 1.72 → 0 | 1.9 → 0 | (was CV) |
| HEX graph net hex_pre46_s5 (gnn:hex dropped) | removed | 25 → 0 | 0.013 → 0 | ~1.8 → 0 (25 of 125 nets) | (was HEX) |
| **total** | | **769 → 749** | **51.94 → 53.64** | **80.8 → ~81** | |

- With the savings (base weights −1.90, duplicate `cpsam_v2` −1.23, fp16-nodet −21.42 on 42.85 GB of regression-member weights), the deploy is **≈ 29.1 GB**.
- Updated per-group table for the current stacks:

| member group | files | GB (fp32) | steady s | feeds |
|---|---|---|---|---|
| ViT-L s46+s47 (e8-11 single + final TTA4) | 10 | 12.13 | 8.4 | CD |
| recipe A s46+s47 e8-19 (A_cd half) | 24 | 8.24 | 11.3 | CD |
| recipe A s46+s47 e20 TTA4 | 2 | 0.69 | 1.7 + 0.2 tokens | CV, HEX, CD (A_cd half), CV frozen heads, detector frozen-head encoder |
| P1 five members + CPSAM backbone | 6 | 7.33 | 9.6 | CD, CV, HEX |
| CPSAM 512 cps5 e8-12 | 5 | 6.09 | 4.8 | HEX |
| DINOv3-B D s46b + s47 e16-20 TTA4 | 10 | 3.43 | ~4.4 | CD; s46b also CV |
| DINOv3-B CV s47 + s49 (lane_cv) e16-20 | 10 | 3.43 | 5.3 | CV |
| cvw3 s46 + s47 e8-12 | 10 | 3.43 | ~3.4 | HEX |
| CPSAM segmentation `cpsam_v2` | 1 | 1.23 (same bytes as the P1 backbone) | 1.7 | CD (mask), HEX (seg features, cell-route points) |
| pre-only detectors s46/47/48 | 3 | 1.03 | 1.9 | HEX, CV |
| 9 full + 4 frozen-head detectors | 13 | 3.09 | 6.9 | HEX |
| hex6 corrector (12 detectors, 8 shared) | 14 | 4.12 (1.37 not shared) | 1.5 | HEX |
| HEX features + 100 HEX graph nets | 100 | 0.05 | ~19.1 (features 11.5 + nets ~7.4) | HEX |
| CV graph nets + sim nets + frozen heads | 263 | 0.20 | 1.2 | CV, HEX |
| cell-route features | 0 | 0 | 0.3 | HEX |
| base weights (not needed, verified) | 4 | 1.90 | – | none |
| code, params, small files | ~280 | 0.01 | – | – |
| **total** | **749** | **53.64** | **~81** | |

### 1c. 39-column HEX ridge with the attention-pool member (11-11 afternoon; measured, `out/budget_fp32_v3.json`, `budget_analysis_v3.json`)

| member group | change | files | GB (fp32) | steady s | first s | feeds |
|---|---|---|---|---|---|---|
| attention-pool DINOv3-B `attnp5` (backbone lane f99 s46, raw e8-11 + final, single view) | added | 5 | 1.77 | 2.0 | 3.2 | HEX |
| HEX graph net hex_pre46_s5 (the joint ridge reads g_hex46) | back | 25 | 0.013 | in HEX graph nets | | HEX |
| **total** | | **779** | **55.42** | **98.1** (shared GPU 3) | **166** | |

- **Not read:** the AttnPool build's SSL checkpoint. Its weights are overwritten by the strict loads, and outputs with and without it are bit-identical (`test_attnp_f99.py`).
- **Size with all savings:** 55.42 − 1.90 (base weights) − 1.23 (duplicate `cpsam_v2`) − 22.31 (fp16 copies: 44.62 GB → 22.31 GB) = **≈ 30.0 GB**.
- **Steady stages (v3), seconds:**

| stage | s | stage | s |
|---|---|---|---|
| HEX features + 125 graph nets (build 10.2 + nets 21.3) | 31.4 | DINOv3-B CV | 5.6 |
| recipe A A_cd | 12.1 | D (two seeds) | 4.7 |
| P1 | 9.7 | cvw3 (two seeds) | 3.9 |
| ViT-L | 9.0 | attnp5 | 2.0 |
| 13 detectors | 6.6 | pre-only detectors | 1.9 |
| cps5 | 5.7 | A | 1.7 |
| segmentation | 1.7 | hex6 | 1.4 |
| CV parts | 1.0 | cell-route | 0.3 |

- **Graph nets:** they grew from 9.2 s (125 nets, 11-10) to 17.1 s (100 nets) to 21.3 s (125 nets) on the same code. CPU contention on the main pod is my guess; I have not measured it.
- **Determinism:** the fp32 run reproduces the deploy run 20f (CD identical, HEX and CV within 8e-9).

## 2. Half-precision storage

Storage variants are simulated at load time; no files are written:
- **fp16:** every float tensor with more than 16 elements, cast to fp16 and back. Metadata such as log_mean and log_std stays fp32.
- **bf16mix:** tensors with ≥ 2 dims in bf16, 1-D tensors (norm weights, biases) kept in fp32. Under GPU bf16 autocast, matmul weights are rounded to bf16 anyway.
- The deploy form would store the half file and call `.float()` at load. Inference then runs exactly as in the simulation: the same fp32 module and the same autocast.

Coverage: every weight file goes through the cast hooks. The one exception is the DINOv3-B base `model.safetensors` (read by `safe_open`), which is overwritten by the strict load.

### 2a. Result: half precision for everything fails the rule for CV and HEX

Relative change per column against the fp32 run, on the same 20 frames (percent):

| variant | GB cast | CD median / max | CV median / max | HEX median / max |
|---|---|---|---|---|
| fp16, all float tensors | 51.6 | 0.013 / 0.032 | 0.168 / 0.49 | 0.381 / 1.83 |
| bf16mix | 51.5 | 0.006 / 0.045 | 0.158 / 0.84 | 0.541 / 2.16 |

- **The rule (median < 0.05 %):** CD passes under both variants. CV and HEX fail under both, by a factor of 3-11.
- **The regression members barely move.** Change of each member's own output under fp16, median / max:

  | member | median / max |
  |---|---|
  | ViT-L | 0.023 / 0.14 % |
  | recipe A, A_cd | 0.022 / 0.09 % |
  | recipe A, A | 0.023 / 0.14 % |
  | P1 | 0.003 / 0.02 % |
  | D | 0.028 / 0.10 % |
  | cps5 | 0.037 / 0.21 % |
  | DINOv3-B CV | 0.037 / 0.13 % |
  | order-DINOv3 | 0.051 / 0.27 % |
  | cvw3 | 0.031 / 0.19 % |

  bf16mix gives the same magnitudes.
- **The peak-based parts move much more:**
  - The number of detector peaks changes by up to 15 per frame: median 0.3 %, max 2.0 %.
  - The hex6 feature f6 changes by a median of 1.8 % (max 4.2 %).
  - The graph-net outputs built on the peaks (graph, g_hex46, g_s10x3, best, graph_cv) change by up to 0.01-0.03 in log units.
  - The cell-route features change too: median 1.1 %.
- **Why:** peaks are thresholded local maxima of the detector heat maps, so a tiny heat-map change adds or drops a peak, and every feature built on the peak set moves with it.
  - The CPSAM segmentation mask is thresholded in the same way.
  - The weights that matter are therefore the detectors' and the segmentation backbone's, not the regression members'.
- **Timing:** the fp16 and bf16mix wall times (92-94 s steady) are not comparable with fp32, because the CV lane's DINOv3 build shared GPU 3 during those runs. The time saved by half storage is the read and transfer of half the bytes.

### 2b. Selective half precision: fp16 for every regression member passes on all three columns

Both variants use fp16, which keeps 10 mantissa bits against bf16's 7. Each ran on the same 20 frames through the GPU 3 guard (`run_budget2_20261111.sh`, 05:35-06:38 UTC).

| variant | cast to half | kept fp32 | GB cast | saved |
|---|---|---|---|---|
| fp16-nodet | ViT-L, recipe A e8-19, P1 seeds, cps5, D, DINOv3-B CV, order-DINOv3, cvw3 | the 16 detectors and hex6 (5.5 GB), detector frozen heads and their recipe A e20 encoder (0.69 GB), `cpsam_v2` (1.22 GB), HEX and CV graph models (0.26 GB) | 42.7, of which 1.55 is base weights that are dropped anyway | 20.6 GB |
| fp16-cdonly | ViT-L, recipe A e8-19 | everything else | 20.4 | 10.2 GB |

Relative change against fp32 per column, median / max (percent):

| variant | CD | CV | HEX | rule (median < 0.05 %) |
|---|---|---|---|---|
| fp16 (everything, section 2a) | 0.013 / 0.032 | 0.168 / 0.49 | 0.381 / 1.83 | fails CV, HEX |
| **fp16-nodet** | **0.008 / 0.029** | **0.005 / 0.024** | **0.009 / 0.052** | **passes all three** |
| fp16-cdonly | 0.007 / 0.023 | 3e-7 / 1e-6 | 8e-8 / 4e-7 | passes all three |

- **fp16-nodet is the default under the rule.**
  - It halves every regression member and saves 20.6 GB.
  - Its medians are 6-10 times below the 0.05 % threshold on all three columns.
  - Its HEX max (0.052 %) is 35 times smaller than with fp16 for everything.
- **The detectors were the cause.** With them in fp32, these are bit-identical to fp32 on all 20 frames: the peak counts, the hex6 feature f6, the cell-route features, the HEX and CV graph features, and the ridge and quartet parts.
  - The largest part changes left are the members' own outputs: median up to 0.051 %, max up to 0.27 % (both order-DINOv3).
  - The columns move less than any single member, because each column averages several members.
- **fp16-cdonly:** CV and HEX differ only at 1e-9, which is the fp32 run-to-run noise level (CV within 8e-9). The CD-only groups do not touch the other columns.
- **Deploy form:**
  - A one-off conversion writes half copies of the nodet files.
  - The entry calls `.float()` on every tensor at load, so inference is exactly what was simulated.
  - Not done: building nothing, and writing about 20 GB of new files needs the coordinator's OK.
- **Runtime:** the half-stored groups need half the bytes read, and half the host-to-GPU transfer if the cast to fp32 happens on the GPU. This was not measured as wall time, because the simulation casts back to fp32 on the CPU and GPU 3 was shared during these runs.

### 2b'. fp16-nodet re-measured on the current stacks (11-11 14:07-14:38 UTC, `out/budget_fp16-nodet_v2.json`)

| column | median | max |
|---|---|---|
| CD | 0.009 % | 0.027 % |
| CV | 0.005 % | 0.021 % |
| HEX | 0.006 % | 0.030 % |

- 44.4 GB cast (42.85 GB of non-base files), and 8.86 GB kept in fp32.
- This passes the coordinator's rule (median < 0.05 % per column), so the fp16 copies were written (section 2e).

### 2e. fp16 copies written (11-11 14:41-14:5x UTC; coordinator approval conditional on 2b', which passed)

- **Script:** `inference/docker/convert_fp16.py 6`, CPU pod. It has a gate: it reads `budget_analysis_v2.json` and exits unless every fp16-nodet median is < 0.05 %.
- **Files:** the 77 files the fp16-nodet run read and cast, minus the base weights:
  - P1 seeds;
  - recipe A e8-19 for both seeds;
  - D s46b + s47;
  - ViT-L s46 + s47;
  - DINOv3-B CV s47 + s49;
  - cvw3 s46 + s47;
  - cps5;
  - 3 sim nets.
- **Size:** 42.85 GB in, **21.43 GB out**, at `lane_cellroute_20261109/deploy_fp16/<original relative path>`. Nothing was overwritten.
- **Check:** every copy was reloaded and compared bit for bit with original → fp16 → original dtype, exactly what the parity runs simulated. 0 mismatches, 0 errors. Result in `inference/docker/convert_fp16_result.json`.
- **Not done (packaging step, waits for the PI's word):**
  - Point the entry at the copies (path map plus `.float()` at load).
  - Skip the base-weight reads.
  - Build the image.

### 2f. With the attention-pool member (v3, 11-11 17:53-18:27 UTC)

- **fp16-nodet against fp32, 20 deploy frames, median / max:** CD 0.009 / 0.027 %, CV 0.005 / 0.021 %, HEX 0.007 / 0.026 %. Passes.
- **New copies:** the 5 attnp5 checkpoints, 1.769 GB → 0.885 GB, bit-identical to the simulated cast (`convert_fp16.py 4 --suffix _v3`, `convert_fp16_result_v3.json`). The 77 existing copies were skipped.
- **`deploy_fp16/` now:** 82 files, 22.31 GB.

### 2c. Size after the no-OOF-cost savings

| step | GB |
|---|---|
| fp32 census (section 1) | 51.94 |
| − base weights (verified unused) | −1.90 |
| − duplicate `cpsam_v2` | −1.23 |
| − fp16-nodet (half of 41.13 GB of regression-member weights) | −20.57 |
| **total** | **≈ 28.2** |
| − order-DINOv3, dropped by the CV lane's 6-component freeze (11-11 07:54 UTC; half of 1.72 GB) | −0.86 |
| **total with the current CV stack** | **≈ 27.3** |
| + D second seed: CD lane seed audit, full-data DINOv3-B s47 epochs 16-20 TTA4 (half of 1.72 GB) | +0.86 |
| **total with the current CV and CD stacks (11-11 05:35 EST)** | **≈ 28.2** |

- **D second seed:** it also doubles D's time, from 2.2 s to about 4.4 s per frame (estimate: 20 more forwards).
- The CD lane's s47 is a different run from the CV lane's DINOv3-B s47 (`lane_cv_20261108/arms/dinov3_b16_f99_s47`, single view for dino_logcv). The two share no files.

- The entry runs the order-DINOv3 member only when the CV weights use it. Under the new CV params it is never opened, and that also saves its 1.9 s per frame.

- **What stays fp32:** detectors and hex6 5.5 GB, `cpsam_v2` 1.2 GB, recipe A e20 0.7 GB, graph models and heads 0.3 GB.
- **Effect on section 3:** every member in its table is stored in half under this default, so each GB-saved figure there halves.
  - Example: dropping the ViT-L single-view files saves 4.85 GB, not 9.7 GB.

## 3. Snapshot rules: what the lanes validated and what each rule costs

- **What the lanes validated is prediction averaging.** Every multi-file member averages log predictions over epochs, flips and seeds. This is shown in:
  - `lane_cd_20261108/source/cd_read_20261108.py`, which reads the per-epoch `predictions_epoch_XX.npz` and `predictions_final_tta.npz` files and never averages weights;
  - the HEX family reader;
  - the recipe A reads.
- **My entry follows the same rule:** `members.read_logs` takes the mean of standardised log outputs per checkpoint.
- **Averaging weights into one file is a different model with no OOF yet**, with one exception: the ViT-L runs trained weight-averaged tracks (swa6 from epoch 6, ema998, ema999) as separate runs.
  - Every ViT-L fold run, s46 f0-f4 and s47 f0-f4, has `swa6/runs/foldK/dinov3_l16t/final.pt` with `predictions_final_tta.npz`.
  - So do both full-data seeds.
  - The OOF of "one SWA file per seed, TTA4" is therefore readable from stored predictions, with no training.

| member | validated rule | files | GB | cheaper rule to price (from stored predictions) | GB saved |
|---|---|---|---|---|---|
| ViT-L (CD) | per seed 0.5 · mean log single view e8-11 + 0.5 · log TTA4 final; geometric mean of s46 and s47 | 10 | 12.13 | final.pt TTA4 only, 2 seeds | 9.70 |
| | | | | swa6 final.pt TTA4, 2 seeds (already trained, OOF stored) | 9.70 |
| | | | | one seed (as validated) | 6.07 |
| recipe A, A_cd (CD) | per seed 0.5 · log TTA4 e20 + 0.5 · mean log single e8-19; 2 seeds | 24 + shared e20 | 8.24 | e20 TTA4 only (the A rule) | 8.24 |
| | | | | single e16-19 instead of e8-19 | 5.49 |
| P1 (CD, CV, HEX) | 5 seeds, mean × OOF scale | 5 + backbone | 6.10 (+1.23 backbone) | 3 seeds / 1 seed | 2.44 / 4.88 |
| CPSAM 512 cps5 (HEX) | exp mean log e8-12, single view | 5 | 6.09 | e12 only | 4.87 |
| DINOv3-B CV (CV) | per seed mean log e16-20; 3 seeds, log mean | 15 (5 shared with D) | 5.15 | seed s46b only (shared with D) | 3.43 |
| | | | | e20 only per seed | 4.12 |
| DINOv3-B D (CD) | mean log over 4 flips × e16-20 | 5 | 1.72 | e20 TTA4 only | 1.37 |
| order-DINOv3 (CV) | mean log e16-20 | 5 | 1.72 | e20 only | 1.37 |
| cvw3 (HEX) | mean log e8-12 | 5 | 1.72 | e12 only | 1.37 |

- **Fewer files also cut runtime almost in proportion**, because the load share dominates. Examples:
  - ViT-L final.pt only: 7.6 s of load becomes about 1.5 s (estimate).
  - recipe A without e8-19: about 11 s saved (estimate).

## 4. Runtime: where the time per frame goes

**Steady frame: 80.8 s median.** It breaks down as follows:

| stage | seconds |
|---|---|
| HEX features + HEX graph nets (11.5 CPU feature build + 9.2 graph nets) | 20.9 |
| recipe A A_cd | 11.3 |
| P1 | 9.6 |
| ViT-L | 8.4 |
| 13 detectors | 6.9 |
| DINOv3-B CV | 5.3 |
| cps5 | 4.8 |
| D | 2.2 |
| pre-only detectors | 1.9 |
| order-DINOv3 | 1.9 |
| A | 1.7 |
| cvw3 | 1.7 |
| segmentation | 1.7 |
| hex6 | 1.5 |
| CV parts | 1.2 |
| cell-route | 0.3 |

- Across the regression members, weight loading takes 29.6 s against 7.7 s of forward passes.
- The earlier deploy run's 74 s was the same pipeline, with cps5 stubbed and less contention on GPU 3.

**First frame: 159 s.** Of this:
- reading 51.6 GB of weights took 29 s, from page cache;
- the rest is model builds and first use: the cellpose model, the P1 net, 16 detector builds (each CPDINO build also reads the 0.34 GB base file), hex6, graph nets, and CUDA and cuDNN start-up.

A Grand Challenge job handles one image (`recipe_infer.main` reads one file), so **each job costs about the first-frame time**. On top of that come container start and any cold-disk penalty, neither measured here.

**GPU alone (estimate, not measured: no GPU was idle).**
- GPU kernels make up a small part of the frame: 7.7 s of member forwards plus parts of P1, the detectors, segmentation and hex6.
- The backbone lane measured time-slicing at about 2× for two jobs on one H200.
- So an unshared GPU saves about 4-8 s: roughly 73-77 s steady and about 150 s for the first frame.

**Frames batched (estimate).**
- Batching across frames is not possible inside Grand Challenge, because there is one image per job.
- If one process handled many frames with members kept on the GPU instead of re-loaded, the steady frame would drop to about 81 − 29.6 ≈ 51 s.
- Running each member's 4 flips as one batch of 4 would cut forwards further. It changes bf16 numerics slightly (the batch-shape effect seen with the detectors), so it needs a parity check first.

### 4b. Re-measure on the current stacks (11-11 13:35-14:38 UTC, `out/budget_fp32_v2.json`, `budget_analysis_v2.json`)

The current stacks are the CD two-seed D, CV 6-component, HEX seedaudit10 with the refit cell-route step, and the real cps5 f99. 20 deploy frames on GPU 3, which was shared with other lanes' jobs.
- **Wall time:** steady 90.0 s median (sum of stages 90.9 s); first frame 213 s. Peak GPU memory 17.0 GB.
- **First frame:** reading 53.3 GB took 43.7 s, against 29 s before. The newly added files were not in the page cache.
- **Determinism:** the fp32 run reproduces the no-stub deploy run 20d. CD is identical on 20/20 frames, HEX within 4e-9, CV within 1e-8.

| stage | steady s | before (section 4) |
|---|---|---|
| HEX features + 100 HEX graph nets (build 9.7 + nets 17.1) | 26.9 | 20.9 (11.5 + 9.2, 125 nets) |
| recipe A A_cd | 11.7 | 11.3 |
| P1 | 9.7 | 9.6 |
| ViT-L | 8.6 | 8.4 |
| 13 detectors | 6.6 | 6.9 |
| cps5 | 5.6 | 4.8 |
| DINOv3-B CV | 5.5 | 5.3 |
| D (two seeds) | 4.6 | 2.2 (one seed) |
| cvw3 (two seeds) | 3.9 | 1.7 (one seed) |
| pre-only detectors | 1.9 | 1.9 |
| A | 1.7 | 1.7 |
| segmentation | 1.6 | 1.7 |
| hex6 | 1.4 | 1.5 |
| CV parts | 1.0 | 1.2 |
| cell-route | 0.3 | 0.3 |
| order-DINOv3 | 0 (not run) | 1.9 |

- **Load against forward:** weight loading for the regression members is 34.7 s, against 8.0 s of forwards.
- **Graph nets:** the HEX graph nets took 17.1 s for 100 nets, against 9.2 s for 125 nets yesterday. They are now the largest single stage.
  - I have not explained this. The nets load 100 small files and run on 4 CPU threads, so CPU contention on the main pod is my guess; I did not measure it.
  - It is the first place to look for a per-job cut.
- **Estimates as before:** the GPU alone saves about 4-8 s. Members kept on the GPU (not possible with one image per job) would save the 34.7 s of loading.

**Per-job cuts with no change to validated numbers:**
- Half-precision storage, if section 2 passes: half the bytes to read and move.
- Fewer snapshot files, if the lanes accept the OOF price: time drops almost in proportion.
- HEX feature build with more than 1 CPU worker (11.5 s at workers=1).
- Removing the duplicate CPSAM file and the base-weight reads.
