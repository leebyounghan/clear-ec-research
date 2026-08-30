# Reproduce / continue on a new GPU

## 0. Hardware & runtime notes

- Developed on a single **RTX 4090 (24 GB)**. ViT-L at 256-crops/batch-16 fits **only with gradient
  checkpointing** (already in the train scripts). A bigger GPU (A100/H100 80 GB) removes that constraint and
  unlocks larger crops — see `NEXT_STEPS.md`.
- Always run with `PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  (`PYTHONNOUSERSITE=1` avoids a broken `~/.local` urllib3/transformers shadow.)

## 1. Environment

Python 3.10. Core versions actually used:

```
torch 2.13.0+cu130      numpy 1.26.4         cellpose 4.2.1.1
opencv 4.11.0           scikit-image 0.25.2  scipy 1.15.3
SimpleITK 2.5.5         transformers 5.14.1  (transformers only for the DINOv2 experiment)
```

See `env/requirements-clearec.txt`. Cellpose provides the `CellposeModel` wrapper + the `cpsam_v2` ViT-L
weights (place at `cpsam_models/cpsam_v2`). The submission container pins its own minimal set
(`submission/requirements.txt`): cellpose 4.2.1.1, numpy 1.26.4, scikit-image 0.24.0, scipy 1.13.1,
opencv-headless 4.10, SimpleITK 2.4.0.

## 2. Data

Download from the challenge (`train_mha_00.zip … _08.zip` = 9000 `.mha`, `final_train_ids.csv` = labels,
`train_green.zip` = 25 ROI reference tiffs). Layout expected by the scripts (they hardcode
`ROOT="datasets/clear_ec_official"`):

```
datasets/clear_ec_official/
  train_mha/*.mha              9000 raw images (1296×972 uint8)
  final_train_ids.csv          ID, CD, CV, HEX, slide_id
  green_eval/                  25 reference images (mha/ + annotations.json)
```

## 3. Build the caches (memmaps the training scripts read)

Order matters; each writes into `datasets/clear_ec_official/`:

```bash
python code/cache_native.py          # img_native_512.dat + cache_native_meta_512.npz  (2.2G) — REQUIRED
python code/precompute_clahe.py      # img_clahe_512.dat        (2.2G) — for 'full' CLAHE members
python code/precompute_full_clahe.py # img_full_clahe.dat       (11G)  — full-frame CLAHE
# img_full_raw.dat (11G) is built inline by train_fullframe.py's helper; or replicate cache_native at full size
```

Then the label-engineering artifacts:

```bash
# slide-averaged targets (adopted). Build both:
#   Y_slidesmooth.npy      — averages only over TRAIN frames (for held-out A/B, no leakage)
#   Y_slidesmooth9000.npy  — averages over ALL frames (for the all-data final)
python code/build_slidesmooth.py     # (see NEXT_STEPS if missing; logic is in train_slidesmooth.py history)
```

## 4. Reproduce the current best (held-out, measurable)

Train the four ensemble members on the 7000-train split, select on val, report test:

```bash
# s2 (best single), attn, long, HEX-specialist
CLAHE=1 TAG=clahe_s2   python code/train_finetune.py
CLAHE=1 TAG=attn_clahe python code/train_stdpool.py      # attn pooling variant
CLAHE=1 TAG=clahe_long EPOCHS=40 python code/train_finetune.py
METRIC=2 TAG=spec_hex  python code/train_specialist.py
# cache member preds (val+test, 40-view TTA) then per-metric greedy ensemble:
python code/cache_member_preds.py
python code/ens_specialist.py          # -> test 8.92 ; results/ens_specialist_results.json
```

## 5. Reproduce the final submission (`v4`, all-9000, unmeasurable)

```bash
# retrain each member on ALL data with slide-averaged targets:
YSRC=smooth9000 ARCH=scalar POOL=mean EPOCHS=25 TAG=s2   python code/train_final9000.py
YSRC=smooth9000 ARCH=scalar POOL=attn EPOCHS=25 TAG=at   python code/train_final9000.py
YSRC=smooth9000 ARCH=scalar POOL=mean EPOCHS=40 TAG=cl   python code/train_final9000.py
YSRC=smooth9000 ARCH=spec  METRIC=2  EPOCHS=25 TAG=xHEX  python code/train_final9000.py
# stage weights, build image, smoke-test, export tarball:
bash code/finalize_container.sh        # -> clear-ec-algo-image-v4.tar.gz (~8.8G)
```

Model selection is impossible without a val split, so `train_final9000.py` uses fixed epoch counts
(validated on held-out) and saves a checkpoint every 5 epochs as a hedge.

## 6. Verify the container locally (CPU smoke)

```bash
docker build -t clear_ec_algo:v4 clear-ec-algo/     # weights baked into image/model/
docker run --rm --network none -e CUDA_VISIBLE_DEVICES="" \
  -v $SMK/input:/input:ro -v $SMK/output:/output clear_ec_algo:v4
# expect three JSONs with sane values (e.g. CD~2900, CV~0.39, HEX~0.52 on 0001-19)
```

`code/verify_container_infer.py` runs the container's `predict()` on held-out val to confirm it reproduces
the expected member numbers.

## Script map (65 files in `code/`)

- **Caching**: `cache_native.py`, `cache_images.py`, `precompute_clahe.py`, `precompute_full_clahe.py`,
  `cache_member_preds.py`, `extract_features.py`
- **Train (regression)**: `train_finetune.py` (base), `train_stdpool.py`/`train_multitask.py`/`train_dldl.py`
  (variants), `train_specialist.py` (per-metric), `train_fullframe.py` (rejected), `train_slidesmooth.py`
  (A/B), `train_final9000.py` (all-data final), `train_dino.py` (rejected)
- **Ensemble/stack**: `ens_specialist.py` (main), `ens_final.py`/`ens_two.py`/`ens_three.py`/`ens_tta.py`,
  `stack_fft_test.py`, `fusion_metalearn.py`
- **Analysis (the important negatives)**: `eval_dino_ensemble.py`, `roi_ceiling_test.py`,
  `roi_region_diag.py`, `green_roi_recovery.py`, `green_marker_seeded.py`, `fft_morphometry2.py`
- **Segmentation/SSL attempts (rejected)**: `finetune_cpsam_pseudo.py`, `round_*.py`, `train_pointdet*.py`,
  `train_unet.py`, `train_semisup*.py`, `mae_pretrain.py`, `micro_sam_finetune.py`
- **Container**: `finalize_container.sh`, `verify_container_infer.py`
