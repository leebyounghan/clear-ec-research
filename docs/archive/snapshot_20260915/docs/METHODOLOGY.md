> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../METHODOLOGY.md) for the release-facing summary.

# Methodology

## The task & metric

Per grayscale specular-microscopy image (`.mha`, 1296×972, 0.7716 µm/px), predict three scalars:

| Metric | Definition (challenge) | Range |
|---|---|---|
| **CD** | cell density = `N_cells · 1e6 / Σ area_µm²` (cells/mm²) | ~1500–4000 |
| **CV** | `std(area) / mean(area)` of cell areas (polymegathism) | ~0.2–0.6 |
| **HEX** | mean per-cell **IoU with a fitted oriented hexagon** (NOT 6-neighbor fraction) | ~0.4–0.7 |

**Score** = mean of the three per-image MAPEs, averaged over images. Lower is better.

**Critical detail — the hidden ROI.** Ground truth was NOT computed on the whole frame. Annotators picked
a small (~0.05 mm², ~125 cells) well-focused patch **anywhere in the frame** and computed CD/CV/HEX there.
For all but 25 reference images (`train_green/`, which show the box + counted cells), the ROI box is
**not provided**. The baseline's grader restricts metrics to a deterministic per-image 40%×40% random crop,
but the *training* GT came from the annotator's hand-picked ROI. This hidden ROI is the dominant source of
CV error — see [`FINDINGS.md`](FINDINGS.md).

Pixel-to-micron: `p2u = 1000/1296 ≈ 0.7716`. Green reference tiffs are 720×540 (main ÷ 1.8).

## Pipeline (winning approach: direct regression)

```
.mha → central-512 native crop → per-crop CLAHE + 1–99% norm → cpsam ViT-L encoder
     → LayerNorm(pooled features) → 3 per-metric MLP heads → log-space scalar → exp-decode
     → C5 crops × 8 dihedral = 40-view TTA (mean) → per-metric ensemble
```

- **Backbone**: Cellpose-SAM (`cpsam_v2`) ViT-L **encoder only** (segmentation decoder dropped), fine-tuned.
  Gradient checkpointing on each ViT block to fit 256-crops at batch 16 on 24 GB.
- **Input scale**: 256×256 native-resolution crops (cells ≈ 27 px = the scale cpsam expects). We crop the
  **central 512** of the frame and sample 256 windows inside it.
- **Targets**: log-space, normalized by train-split mean/std. Constants in `results/norm_consts.json`:
  - `LM = [7.91992, -0.98248, -0.62114]` (log-mean per metric)
  - `LS = [0.18972, 0.18814, 0.27489]` (log-std)
  - `YMEAN = [2795.84, 0.38114, 0.54599]` (used by SpecNet specialists, which regress in mean-normalized space +1.0)
- **Loss**: log-space L1 (robust to the right-skewed targets). Specialists use per-metric relative-error loss.
- **TTA**: 5 crops (`C5 = 4 corners + center`) × 8 dihedral flips/rotations = 40 views, averaged.

## Split (fixed, seed 42)

`perm = np.random.default_rng(42).permutation(N)` where `N = 8954` (rows in `cache_native_meta_512.npz`,
which is the 9000 images minus a few unreadable). Then `test = perm[:1000]`, `val = perm[1000:2000]`,
`train = perm[2000:]`. **All model selection is on `val`; `test` is only ever reported, never tuned.**
2 rows have HEX=0 (undefined log target) and are dropped from training.

## Model variants (ensemble members)

Each is `code/train_*.py`. Class defs are duplicated across scripts (self-contained); the two architectures:

- **ScalarNet** — encoder → pool → LayerNorm → 3 heads, output in **log-space** (`exp(p·LS+LM)`).
  Pooling variants: `mean` (AdaptiveAvgPool), `attn` (learned query cross-attention), `meanstd` (mean⊕std).
- **SpecNet** — same but heads output in **YMEAN space** (`p·YMEAN`) with a `+1.0` residual; trained
  single-metric (`METRIC` env = 0/1/2) as a per-metric **specialist**.

| tag | script | checkpoint | pool | space | notes |
|---|---|---|---|---|---|
| `s2`  | train_finetune.py (CLAHE=1) | regress_ft_best_clahe_s2.pt   | mean | log | the single best member |
| `at`  | train_stdpool/attn          | regress_ft_best_attn_clahe.pt | attn | log | attention pooling |
| `cl`  | train_finetune.py (long)    | regress_ft_best_clahe_long.pt | mean | log | 40-epoch |
| `xHEX`| train_specialist.py METRIC=2| regress_ft_best_spec_hex.pt   | mean | ymean | HEX specialist |
| `ff`  | train_fullframe.py          | regress_ft_best_fullframe.pt  | mean | log | **rejected** (test 9.18) |
| `dino`| train_dino.py               | regress_ft_best_dino.pt       | cls  | log | **rejected** (corr 0.99) |

## Ensemble (per-metric greedy selection)

`code/ens_specialist.py` (and the faster `code/stack_fft_test.py` once member preds are cached). For each
metric independently, greedily add the member subset minimizing **val** MAPE. Result (measured on held-out
test):

| metric | members | test MAPE |
|---|---|---|
| CD  | `s2 + at`            | 6.41 |
| CV  | `s2`                | 10.39 |
| HEX | `at + xHEX + cl`    | 9.97 |
| **OV** | | **8.92** |

Then a **multiplicative calibration** (β=−1 shading, Gneiting 2011): multiply CV by 0.991, HEX by 0.993
(CD left at 1.0 — its correction hurt on test transfer). Net: **8.89**. This is baked into
`submission/inference.py` (`CALIB` array).

## Final submission model (`v4`)

The two adopted improvements applied and retrained on **all 9000** (no holdout — the leaderboard test is
disjoint, so this is legitimate):

1. **Slide-averaged targets** — GT of frames from the same `slide_id` averaged together (label denoising;
   intra-slide GT rel-std is CD 2.5% / CV 5.3% / HEX 5.5%). `Y_slidesmooth9000.npy`. A/B on held-out showed
   **−0.20 test OV** vs raw targets.
2. **CV/HEX calibration** (above).

Members retrained via `code/train_final9000.py` (env-configurable ARCH/POOL/YSRC/METRIC) →
`final9000_{s2,at,cl,xHEX}.pt` → staged into the container by `code/finalize_container.sh`.

**Caveat**: because v4 trains on all data (including the internal val/test), it **cannot be measured
internally** — the only true score is the leaderboard. To measure honestly, use k-fold OOF (see NEXT_STEPS).

## Submission container

`submission/inference.py` reads `/input/images/corneal-specular-microscopy-image/*.mha`, takes the central
512, runs the per-metric ensemble with 40-view TTA, writes `/output/{cell-density,coefficient-of-variation,
hexagonality}.json` (one float each). Weights load from a mounted GC Model at `/opt/ml/model` if present,
else from `/opt/app/model` baked into the image. bf16 autocast falls back to **fp16 on the challenge T4**
(sm75 has no bf16). Deploy by tagging a release on the `clear-ec-algo` repo (CodeBuild) **or** uploading the
exported `docker save | pigz` tarball directly to the Containers tab.
