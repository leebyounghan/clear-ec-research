> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../ARTIFACTS.md) for the release-facing summary.

# Artifacts — large files (not in git) and how to get them

## September 15 version map

See [STATUS_20260915.md](STATUS_20260915.md) for the delivered sixth-only SSL
archive hash and separate completed all-five SSL checkpoint paths. The latest
archive is `model_full_quality77_ssl_raw_20260914.tar.gz`; it is not the older
September 10 model, and the all-five retraining is not packaged into it.
Local archive delivery does not provide evaluator-accessible external hosting.

## Sep3/4 2026: original archive now available

The user provided `clear-ec-train9000-raw.tar.gz` (6,633,443,603 bytes). The verified
pod copy is `/workspace/artifacts/clear-ec-train9000-raw.tar.gz`; raw images and
separate full-frame caches are in `datasets/clear_ec_fullframe_20260904`.
See [SUBMISSION_20260904.md](SUBMISSION_20260904.md) for checksum, validation,
ID reconciliation, current experiments and submission restrictions. The original
central512 cache was not replaced. Treat the v4 instructions below as historical.

Git holds **code + docs + small result JSONs only**. Weights and image caches are large and are excluded via
`.gitignore`. To continue on a new machine you either **regenerate** (from the public challenge data) or
**transfer** the files below.

## What to transfer vs regenerate

| Artifact | Size | Get it by | Priority |
|---|---|---|---|
| `cpsam_models/cpsam_v2` | 1.2 G | cellpose (`CellposeModel(pretrained_model="cpsam_v2")` downloads it) or copy | **required** |
| `final9000_{s2,at,cl,xHEX}.pt` | 4×1.2 G | **transfer** (or retrain via `train_final9000.py`, ~20 GPU-h) | **required for v4** |
| `regress_ft_best_{clahe_s2,attn_clahe,clahe_long,spec_hex}.pt` | 4×1.2 G | transfer, or retrain held-out members (~16 GPU-h) | needed for measured ensemble |
| `regress_ft_best_dino.pt` | 1.2 G | transfer, or `train_dino.py` | optional (rejected member) |
| `img_native_512.dat` + `cache_native_meta_512.npz` | 2.2 G | **regenerate**: `python code/cache_native.py` | **required** |
| `img_clahe_512.dat` | 2.2 G | regenerate: `precompute_clahe.py` | for CLAHE members |
| `img_full_clahe.dat`, `img_full_raw.dat` | 2×11 G | regenerate (full-frame; only if revisiting full-res) | optional |
| `pool_edge_u8.dat` (fNLA edges) | 2.4 G | transfer (needs the TF `fnla` env to regenerate) | optional (seg experiments) |
| `Y_slidesmooth.npy`, `Y_slidesmooth9000.npy` | small | regenerate: `python code/build_slidesmooth.py` | **required for v4** |
| `member_preds.npz`, `dino_preds.npz` | small | regenerate: `cache_member_preds.py` | for ensemble selection |
| `norm_consts.json` | tiny | in `results/` | reference |

**Minimum set to reproduce v4 and submit**: `cpsam_v2` + the four `final9000_*.pt` + (regenerate caches +
slidesmooth). Everything else supports further experiments.

## Checksums (md5, first 12 hex) — current best model

```
2a39b64fd658  final9000_s2.pt
0bfe641d2d09  final9000_at.pt
3cfe3dd9ab8d  final9000_cl.pt
6b641864e1c4  final9000_xHEX.pt
```

## Transfer recipe

Bundle the required weights (not the regenerable caches) and copy to the new host:

```bash
# on the current machine (from repo parent that contains datasets/ and cpsam_models/)
tar -cf - \
  cpsam_models/cpsam_v2 \
  datasets/clear_ec_official/final9000_{s2,at,cl,xHEX}.pt \
  datasets/clear_ec_official/regress_ft_best_{clahe_s2,attn_clahe,clahe_long,spec_hex}.pt \
  datasets/clear_ec_official/{cache_native_meta_512.npz,Y_slidesmooth.npy,Y_slidesmooth9000.npy,norm_consts.json,member_preds.npz} \
  | pigz > clear-ec-weights.tar.gz          # ~10 G

# on the new machine
mkdir -p <workdir> && cd <workdir> && tar -I pigz -xf clear-ec-weights.tar.gz
# then regenerate the image caches (need the raw train_mha/):
python code/cache_native.py && python code/precompute_clahe.py && python code/build_slidesmooth.py
```

Adjust the hardcoded `ROOT="datasets/clear_ec_official"` in the scripts if your layout differs.

## The submission container image

`clear-ec-algo-image-v4.tar.gz` (~8.8 G, `docker save | pigz`) is the ready-to-upload artifact — it already
contains all weights baked in. Not in git; transfer directly or rebuild with `code/finalize_container.sh`.
