# Checkpoints

The weights are not stored in git. They are distributed as one archive.

| File | Bytes | sha256 |
|---|---:|---|
| `model_freeze-9687076-p3b_20261114.tar.gz` | 15,056,963,340 | `a51a21581409b9cd7c527d90e8f4df6af67f268f575278a3583c15dfbbe3bf1d` |

**Download:** the link will be added here.

## Placement

Extract the archive into this folder so that its relative paths are kept:

```bash
tar -xzf model_freeze-9687076-p3b_20261114.tar.gz -C checkpoints/
```

Afterwards `checkpoints/cpsam_models/cpsam_v2` and `checkpoints/experiments/...` exist. The folder holds 443 files, 16.25 GB.
Check them against `configs/checkpoint_sha256.txt`:

```bash
python3 scripts/verify_checkpoints.py
```

The container reads this folder read-only at `/opt/ml/model`.

## Contents

| Folder in the archive | Component |
|---|---|
| `cpsam_models/cpsam_v2` | Cellpose-SAM: segmentation and the P1 encoder |
| `experiments/deploy_review_20260905/staged/model` | P1: five CD regressors on the Cellpose-SAM encoder |
| `experiments/fulldata_recipeA_20261028/regression_std_s46, _s47` | recipe A: two ViT-B/8 whole-frame regressors |
| `experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46b` | DINOv3 ViT-B/16 CD regressor |
| `experiments/lane_cd_20261108/arms/vitlfu_elr2e5_f99_s46`, `vitl47fu_elr2e5_f99_s47` | DINOv3 ViT-L/16 CD regressors (encoder LR 2e-5) |
| `experiments/lane_cv_20261108/graph_models`, `frozen_heads_full` | CV graph networks and frozen token-statistic heads |
| `experiments/lane_hex_20261108/deploy` | HEX members, cell detectors and graph networks |
| `experiments/green_track_20261028/deploy` | cell detectors and the hexagonality corrector |
| `experiments/lane_backbone_20261108/runs/attnpool_f99_dinov3b16_s46` | attention-pooled DINOv3 ViT-B/16 (HEX column) |
| `experiments/meta_ceiling_20261103/sweep` | simulation-embedding networks |

The archive also holds the five checkpoints of `experiments/lane_cd_20261108/arms/dinov3_b16_f99_s47`, which p3b does not read
(`configs/model_files.txt` lists the 438 files it reads). They are kept so that the archive is byte-identical to the one that was tested.

Some regression checkpoints are stored in fp16 and are cast back to fp32 at load time; the container finds them through
`src/experiments/lane_cellroute_20261109/inference/docker/fp16_files.txt`.
