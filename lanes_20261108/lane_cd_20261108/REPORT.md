# CD lane (GPU 0), from 2026-11-08

Target: 5-fold OOF CD <= 6.11 (best pipeline 6.264), stretch 6.00, by the 11-11 18:00 EST freeze. Lines in `RESULTS.tsv`. Times Eastern.
Rules since 11-08: keep any CD gain on OOF (settings never chosen on the rows reported); CD changes touch only the CD column; no per-source choices; one point estimate per try, no CIs.

## Where CD stands

| Read | Folds | CD | Change vs best pipeline | CI |
|---|---|---:|---:|---|
| Best pipeline (hybrid B + stretch) | 0-4 | 6.264 | | |
| + DINOv3 B/16 s46, three-weight blend P1/A2/F, nested | 0-4 | 6.217 | -0.047 | -0.068, -0.024 |
| + four-flip TTA on DINOv3 and recipe A epoch-8-19 snapshot | **0-4** | **6.205** | **-0.059** | -0.084, -0.033 |
| + ConvNeXt s46 TTA, separate weight (gets 0.05) | 0-4 | 6.204 | -0.060 | |
| + mask density from the HEX lane's Cellpose-SAM segmentation (w 0.1, frames with >= 50 interior cells) | 0-4 | 6.187 | -0.077 | |
| ViT-L/16 in place of ConvNeXt: P1 0.25 / A2 0.45 / DINOv3 TTA 0.15 / ViT-L 0.15 + mask 0.1 | 0-4 | 6.1775 | -0.086 | |
| ViT-L as the mean of seeds 46 and 47 (freeze candidate v2: P1 0.25 / A 0.45 / DINOv3 0.10 / ViT-L 0.20 + mask 0.1) | **0-4** | **6.1726** | **-0.091** | |
| DINOv3 member as the s46/s47 seed mean (coordinator seed audit 11-11, no seed picked; freeze candidate v3) | **0-4** | **6.1741** | **-0.090** | |
| v3 under strict nesting (P1 SCALE and recipe A scalars refit per outer fold; best pipeline rebuilt the same way: 6.2610) | 0-4 | 6.1743 | -0.087 vs strict best pipeline | |
| From-scratch combination search over 22 five-fold members, fully nested (forward selection / ensemble selection) | 0-4 | 6.1733 / 6.1767 | -0.0008 / +0.0026 vs v3 | |
| + DINOv3 s46 + s47 + ConvNeXt + DINOv2 (stored), cross-fit | 0, 4 | 6.185 (vs 6.269) | -0.084 | -0.130, -0.039 |
| Tier p (deploy CD t1) / p2 (no DINOv3 s47 half), strict | 0-4 | 6.1737 / 6.1723 | | |
| **p3 = p2 with ViT-L s46 retrained at encoder LR 2e-5** (11-14; every fold lower; 10-fold -0.0174 vs tier p) | **0-4** | **6.1608** | -0.0128 vs tier p, -0.0115 vs p2 | -0.023, -0.003 |
| **p3b = p2 with both ViT-L halves retrained at encoder LR 2e-5** (11-14; every fold lower; 10-fold -0.0306 vs tier p, -0.0132 vs p3) | **0-4** | **6.1479** | -0.0258 vs tier p, -0.0129 vs p3 | -0.040, -0.012 |

## What this says

- Other encoder families err differently on CD (error r with B 0.87-0.91). At 20-30 % they lower CD; the five-fold read of one DINOv3 seed confirms it in every fold (gain mostly on new-format IDs, -0.076; old-format -0.017).
- Four-flip TTA over epochs 16-20 lowers a family member's own CD by about 0.05-0.1 and adds -0.008 in the blend.
- ConvNeXt looked strong on folds 0 and 4 (-0.045) but adds -0.002 on five folds once DINOv3 is in: the screen folds were its best. Folds 0+4 can overstate a member; read five folds before stacking more of a family.
- ViT-L/16 (approved 11-07) is the best single member on fold 0 (CD 6.455) and adds -0.012 there on top of DINOv3 + ConvNeXt; folds 1-3 running on GPU 0, folds 0 and 4 from the backbone lane.
- Closed as tested: graph/sim CD column, per-source blend weights, per-source stretch after the blend, detector density as a global component. Dropped under the 11-08 evening charter: density on source-B rows (-0.001, a per-source choice).

## Running

GPU 0 (runner v3 `source/queue_cd_v3_20261108.sh`): ViT-L full-data s46 (SSL + 12 epochs, started 14:54), done 18:54 (4.0 h); GPU 0 since 18:57 belongs to other lanes, except the full-data DINOv3 s47 build (11-11 04:18-05:26, seed audit); since 05:27 the HEX lane's CPSAM seed audit (~14:30). No CD GPU job planned. Full-data s47 built on GPU 3 (backbone lane), 21:37. No CD build pending.

- Mask area is a direct read of density: calibrated mask CD at weight 0.1 lowers CD in all five folds (-0.017 on the stack). Robust area statistics, the densest window and a rotated second segmentation give the same or less.

## Directions, continue or stop (09:00 Eastern daily)

11-08 18:00 (first entry; gap to close about 0.09 on CD):

| Direction | GPU-h so far | Gain so far on CD | Decision |
|---|---:|---|---|
| Other-family members: DINOv3 B/16 | 5 + TTA | five folds -0.047 alone, -0.055 with TTA | continue: seeds 47, 48 queued |
| Other-family members: ConvNeXt B | 5.4 + TTA | five folds -0.002 beyond DINOv3 | stop (seed 47 removed from queue) |
| ViT-L/16 DINOv3 (approved 11-07) | 7.6 (f1, f2) | folds 0-1 -0.036 | continue: f3 running, f4 from backbone lane |
| Flip TTA over kept epochs | 1 | -0.008 | continue: applied to every member |
| Recipe A epoch 8-19 snapshot | 0 (stored) | -0.004 | kept, done |
| CD-weighted loss (x3) | 0 | not read | one test, DINOv3 folds 0 and 4 (~3 GPU-h) |

11-09 09:00 (gap to 6.11 now 0.068; five-fold stack 6.1775):

| Direction | GPU-h so far | Gain so far on CD | Decision |
|---|---:|---|---|
| ViT-L/16 DINOv3 | 11.4 (s46 f1-3) + 4.2 (s47 f0, f4) | five folds -0.009 beyond DINOv3 + mask; s47 adds -0.005 on folds 0, 4 | continue: s47 f1-3 running (done ~12:00); s48 queued, kept only if the s47 five-fold read gains |
| DINOv3 B/16 seeds | 4 (s47 f1-3) + TTA | s47 weight 0 on five folds | stop (s48 removed) |
| Mask density (HEX lane Cellpose-SAM, read-only) | 0 (CPU ~1 h) | -0.017 on five folds | done, kept; robust stats, window, rotated segmentation, Voronoi CD add nothing (closed) |
| Cross-lane members (CV lane DINOv3 cvw3) | 0 | 0 | stop |
| Uncertainty-based MAPE shift | 0 | worse | stop |
| CD-weighted loss (x3), DINOv3 f0/f4 | 0 | not read | one test after ViT-L s47 (~3 GPU-h) |

11-09 evening: distribution head (HL-Gauss) on DINOv3 closed (five folds +0.008, weight 0 next to the ViT-L seed mean); DINOv3 ViT-H+/16 started (PI-approved), folds split over GPUs 0 (f0, f1), 2 (f4, f2, CV lane), 3 (f3, backbone lane), ViT-L recipe with SSL.

11-10 early: ViT-H+/16 closed after folds 0 and 4 (CD -0.0034 averaged into the ViT-L slot; HEX lane +0.029). Recipe A on the 2x technician-window crop (backbone lane) reads -0.0137 on folds 0 and 4 in the CD stack: fold 3 on GPU 0 (06:50 EST), folds 1-2 on GPU 2. Next on GPU 0: the backbone lane's target-denoising fold 4 if it arrives, else full-data DINOv3 s46 and ViT-L s46/s47 for the rehearsal and the 11-12 rebuild.

11-10 09:00 (gap to 6.11: 0.063; five-fold stack 6.1726 = freeze candidate v2):

| Direction | GPU-h so far | Gain so far on CD | Decision |
|---|---:|---|---|
| ViT-L seeds (s46 + s47 mean) | ~27 | -0.0049 for s47 on five folds | done; s48 only as filler after the full-data builds |
| DINOv3 ViT-H+/16 | ~11 (f0) + GPU 2 f4 | folds 0,4: -0.0034 in the ViT-L slot | closed (coordinator; HEX +0.029) |
| Distribution head (DINOv3) | ~7 | five folds +0.008 | closed |
| Recipe A 2x-window (backbone lane member) | 5 (f3 on GPU 0) | five folds: +0.0026 (4 folds had read -0.0041, 2 folds -0.0137) | closed 11-10 15:01 (not in the CD stack) |
| Target denoising fold 4 (backbone lane) | ~7.8 | backbone lane reads it (row-weight done 14:54; soft05 stopped at their request) | done on GPU 0 |
| Full-data builds of the freeze members | 1.9 (DINOv3 done) | - | ViT-L s46, s47 after the denoising pair |

11-11 09:00 (freeze candidate v3 6.1741, -0.090 vs 6.2639; gap to 6.11: 0.064; no open CD GPU direction):

| Direction | GPU-h so far | Gain so far on CD | Decision |
|---|---:|---|---|
| DINOv3 seed audit (coordinator) | 1.1 (full-data s47) | seed mean +0.0016 vs s46 alone (s46 was lucky) | done: seed mean is the member (v3), s47 full data built and verified 05:26 |
| Attention-pool DINOv3-B (backbone lane member) | 0 (theirs) | five folds +0.002 to +0.005 in every form | closed (CD, CV, HEX all weight 0) |
| C-Mixup DINOv3-B (backbone lane member) | 0 (theirs) | folds 0,4 +0.004 to +0.011 | closed |
| Recipe A 2x-window (backbone lane member) | 5 | five folds +0.0026 | closed 11-10 |
| Full-data builds of the freeze members | 7.4 (D s46 1.9, V s46 4.0, D s47 1.1; V s47 on GPU 3) | - | done, all verified (sha256, strict load, held rows reproduced) |
| Deploy-size price list (CPU) | 0 | ViT-L final only +0.002; no ViT-L +0.013; A final only +0.005; P1 + A + mask +0.044 | for the PI size decision; candidate unchanged |

GPU 0: HEX lane CPSAM seed audit until ~14:30 EST (coordinator). The CD lane's remaining work is the deploy side only.
