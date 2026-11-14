# Tier p: T4 seconds per stage (Colab logs), and what the CD-only pieces cost

Deploy lane, 2026-11-14 02:29 EST. Answers the coordinator's 02:16 EST items 1-2. Analysis only: no Colab run, no build, no upload.

## Bottom line

- **The three CD-only pieces cost about 25-28 s per T4 job:**
  - DINOv3-B s47 CD half: about 7.3-8.8 s.
  - ViT-L s46: about 8.7-9.4 s.
  - ViT-L s47: about 8.7-9.4 s.
- **Dropping only the DINOv3 s47 half or only one ViT-L seed saves 6-9 s.**
- **The recipe A replacement should cost +16-18 s, not +20-25 s,** if its two four-flip ViT-B/8 members match the architecture and file size of today's recipe A pair. The current recipe A s46+s47 stage does the same work (2 x 4 ViT-B/8 forwards, 687 MB read) in 15.9-18.2 s. Add about 0.3 s if it is a separate stage.
- **VM 2 is already over 300 s (311-314 s), so dropping models alone does not create margin there.** With the replacement and all three drops, VM 2 lands at about 300-314 s.
- **Code-only fixes would add 15-20 s of margin** (same weights, same forwards, outputs should be unchanged). They remove repeated reads of files the job has already loaded. Each needs a code change, a rebuild and the container test. They raise peak RAM by up to about 3.6 GB over the measured 5.0 GB.

## Sources

The source files are on the pod in `experiments/lane_cellroute_20261109/out/`:

- `colab_t4c6_p_fp16b1.json`: VM 1, frames 0 and 447.
- `colab_b/t4b2_p_fp16b1.json`: VM 2 run b2, frame 0.
- `colab_b/t4b3_p_fp16b1.json`: VM 2 run b3, frames 0 and 447.

What the runs had in common:

- Mode `fp16b1` (the container default on a T4), unloading on, 2 CPU threads.
- Every job finished with rc 0, in recipe mode, on a Tesla T4.
- Stage timers are CUDA-synchronised. Nested calls count in the outer stage (`colab_gc_wrap.py`).
- The rows below add up exactly to each job's wall time.

Model file sizes come from the extracted p model on the build box (`package_20261112/test_model_p`).

## 1. Seconds per stage, one job

- Rows marked "est." split one stage in half. The halves are symmetric: same architecture, same number of checkpoints and flips, same file sizes.
- Each stage includes its own model build, checkpoint reads and copies to the GPU, because unloading clears models after every stage.
- The "model loading" row is the torch.load / safetensors read time summed inside the stages, so it is not added to the wall.

| stage | median | VM1 f0 | VM1 f447 | VM2 run b2 f0 | VM2 run b3 f0 | VM2 run b3 f447 |
|---|---|---|---|---|---|---|
| P1 crop members (5 x central 512 on cpsam_v2) [p1] | 43.9 | 42.2 | 42.0 | 43.9 | 44.8 | 47.7 |
| CPSAM segmentation [seg] | 17.2 | 17.7 | 15.0 | 18.7 | 16.7 | 17.2 |
| recipe A s46+s47 TTA (2 x 4 ViT-B/8 fwd) [recipeA_tta] | 17.7 | 15.9 | 18.0 | 16.7 | 17.7 | 18.2 |
|   of which one seed (est. = half) | 8.9 | 8.0 | 9.0 | 8.4 | 8.9 | 9.1 |
| DINOv3-B dino_cd: s46b + s47 (2 x 5 ep x 4 flips = 40 fwd) [dino_cd] | 15.2 | 14.6 | 14.9 | 16.9 | 15.2 | 17.7 |
|   DINOv3-B s46b CD half, 20 fwd (est. = half) | 7.6 | 7.3 | 7.5 | 8.5 | 7.6 | 8.8 |
|   DINOv3-B s47 CD half, 20 fwd (est. = half) | 7.6 | 7.3 | 7.5 | 8.5 | 7.6 | 8.8 |
| DINOv3-B s46b CV read (5 single fwd, rereads the 5 files) [dino_cv] | 4.0 | 4.2 | 3.6 | 4.5 | 3.8 | 4.0 |
| ViT-L s46+s47 TTA (2 x 4 fwd) [vitl_tta_only] | 18.1 | 17.4 | 18.1 | 17.3 | 18.3 | 18.7 |
|   ViT-L one seed (est. = half) | 9.0 | 8.7 | 9.0 | 8.7 | 9.1 | 9.4 |
| DINOv3-B cvw3 HEX (10 single fwd) [family_mean] | 12.1 | 11.0 | 12.4 | 11.4 | 12.2 | 12.1 |
| DINOv3-B attention-pool HEX (5 single fwd) [attnpool_hex] | 6.9 | 6.7 | 7.0 | 6.5 | 6.9 | 7.4 |
| 3 pre-only detectors, TTA [peaks] | 30.6 | 28.6 | 30.6 | 30.1 | 31.5 | 30.9 |
| 13 other detectors, single view [other_peaks] | 58.3 | 57.0 | 56.4 | 64.0 | 63.9 | 58.3 |
| hex6 corrector (10 detectors; 6 also run in peaks / other_peaks) [hex6] | 43.1 | 41.7 | 41.1 | 50.0 | 47.8 | 43.1 |
| HEX features + HEX graph nets [hex_features] | 4.2 | 4.2 | 3.6 | 5.7 | 3.6 | 5.0 |
| CV graph/sim nets + frozen heads (+1 recipe A s46 token pass) [cv_parts] | 11.8 | 10.7 | 10.4 | 12.3 | 11.8 | 11.8 |
| cell-route features [features] | 1.0 | 1.6 | 0.8 | 1.0 | 1.6 | 0.9 |
| model unload between stages, ~14 calls [_stage_end] | 3.7 | 3.6 | 3.7 | 3.6 | 4.1 | 3.9 |
| preprocess | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| setup: imports + Recipe init [setup_s] | 6.9 | 6.5 | 6.7 | 6.9 | 7.4 | 10.1 |
| rest of the entry: image read, CD/HEX/CV combine, write | 0.3 | 0.3 | 0.3 | 0.2 | 0.3 | 0.3 |
| process start/exit outside the entry | 3.4 | 3.2 | 3.7 | 4.2 | 3.4 | 3.2 |
| WALL per job | 310.7 | 287.4 | 288.4 | 314.2 | 311.1 | 310.7 |
| model loading = torch.load/safetensors read time (inside the stages) | 59.6 | 59.6 | 55.0 | 77.0 | 62.6 | 58.7 |
|   read throughput MB/s | 412.5 | 412.5 | 447.2 | 319.3 | 392.4 | 419.1 |

How the stages map to the manifest (`inference/manifests/deploy_p.json`):

- **dino_cd**: [s46b e16-20] and [s47 e16-20], 4 flips each.
- **dino_cv**: s46b e16-20, single view.
- **vitl_tta_only**: final.pt of vitl16_f99_s46 and vitl16_f99_s47, 4 flips each.
- **family_mean**: cvw3 s46+s47 e08-12.
- **attnpool_hex**: attnp5, 5 checkpoints.
- **other_peaks**: 13 detectors. Four of them are frozen heads, and each frozen head loads the recipe A s46 encoder.
- **hex6**: hex6_blend_params.json lists 10 families: s8 x3, s8multi_s47, sg2_s46-s50 and sg2pre_s46.

The VMs differ mainly in the detector stages (other_peaks +6-7 s, hex6 +2-9 s on VM 2) and in read throughput (319-419 MB/s on VM 2, 412-447 MB/s on VM 1).

## 2. Seconds saved per job without each CD-only piece (load time included)

| piece | saved per job | its own read | basis |
|---|---|---|---|
| DINOv3-B s47 CD half (5 checkpoints, 858 MB, 20 forwards) | 7.3-8.8 (half of dino_cd); at least 6.3-7.9 | 1.9-2.7 s | Estimate: half of dino_cd. The lower bound removes half of the shared net build, which is bounded by dino_cv minus its 858 MB read. |
| ViT-L s46 (606.5 MB, 4 forwards) | 8.7-9.4 (half of vitl_tta_only) | 1.4-1.9 s | Estimate: half of the stage. Slightly high, because the ViT-L net build is shared and the logs do not time it separately. |
| ViT-L s47 | 8.7-9.4 | 1.4-1.9 s | Same as ViT-L s46. |
| both ViT-L seeds | 17.6-19.0 | 2.7-3.8 s | Measured: the whole stage plus its unload call. |
| all three pieces | 24.9-27.8 | 4.6-6.5 s | Half of dino_cd (estimate) plus the whole ViT-L stage. |

Dropping the s47 half leaves dino_cd with s46b only, which is the same s46b that dino_cv reads.

## Margin without dropping models (estimates from bytes and each job's read throughput)

All four items need a code change in the frozen inference code. None has been done.

1. **Skip the dino_cv stage: 3.6-4.5 s (the whole stage).** dino_cv reloads the 5 s46b files that dino_cd just used and repeats their unflipped forward, which is flip 0 of dino_cd's TTA. Keeping dino_cd's flip-0 output would make dino_cv free. It should give identical outputs (same weights, same input, batch 1), but that must be checked.
2. **Read recipe A s46 epoch_20 once instead of six times: 3.8-5.4 s.** It is read by recipeA_tta, by each of the 4 frozen heads in other_peaks, and by cv_parts for the token pass. The 5 extra reads total 1,716 MB.
3. **Don't re-read hex6's six shared detector files: 4.6-6.4 s.** hex6 re-reads sg2_s46-s50 and sg2pre_s46 (2,057 MB), which other_peaks and peaks loaded and dropped minutes earlier. Only the read is counted here. Its forwards may also repeat theirs, but that is not verified.
4. **Read cpsam_v2 once: 2.8-3.9 s.** p1 and seg each read it (1,234 MB).

The total is about 15-20 s. Caching the loaded files in CPU RAM for the stages that share them adds up to about 3.6 GB to the 5.0 GB RAM peak; check the Grand Challenge instance's RAM first. Read time is a lower bound on what each item saves.

## Scripts

- `stage_table.py` and its output are in the pod lane folder `experiments/lane_cellroute_20261109/inference/t4stage_20261114/`. The script reads only the files above.
