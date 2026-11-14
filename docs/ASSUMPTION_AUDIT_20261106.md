# Assumption audit (2026-11-06)

What the project treated as settled, and how strong the evidence really is. Read from git commit `f8bc49b` and the training code. The ranking by score impact is a judgement.

## Readout noise

One fold and one seed cannot resolve most of what they closed. From 8 stored std-member histories: validation mean moves 0.059 on average between consecutive epochs; within-run SD over epochs 11-20 is 0.052 (range 0.028-0.086); seed noise comes on top. Std pooling's fold 4 seed 47 pair (-0.038) would itself have failed the -0.05 gate that closed invariance training at -0.042.

Rule from now on: read the mean of the last five epochs, on at least two folds, paired with a matched reference.

## Beliefs

| # | Belief | Evidence | Not tested | Strength |
|---|---|---|---|---|
| 1 | SSL is at its ceiling; SSL5 suffices | Mean pooling. Fold 3 seed 42: SSL0 9.3017, 1 9.2481, 5 9.2403, 10 9.2445. Fold 4 seed 46: 0 9.1468, 5 9.1333, 10 9.1113, 20 9.1200. Fold 0 seed 46: 5 9.1811, 10 9.2101. SSL seed +0.005, -0.005 | SSL5 is LeJEPA, 5 epochs, LR 1e-5; views differ only by flips, gain 0.9-1.1, offset +-0.03. No crop, blur, noise, optics or style in any SSL code. Only duration and seed varied. SSL pools by mean, regression by std; SSL0/10/20 never read on std | Duration weak; ceiling never tested |
| 2 | Encoder strength is not the limit | Recipe A vs Phase-1 error r 0.91, 0.97, 0.97, one model pair | Indirect: both Cellpose-pretrained ViTs, same labels, CLAHE and flips. No larger encoder under recipe A | Weak |
| 3 | Other encoders add nothing | Added at 20 %: fold 3 -0.014, -0.020; fold 0 -0.012, -0.005; gate -0.02 | No SSL, mean pooling, old cosine schedule, one seed, folds 1, 2, 4 untrained | Weak, unequal budget |
| 4 | Flips-only training is fine | Fold 0 seed 46: -0.042 (CI -0.104, 0.019), CD -0.08, CV -0.07, HEX +0.03; added -0.005 | One bundle (rotation, gamma, blur, noise), same epochs and LR, no second fold or seed, no measured noise or optics model, SSL encoder not re-adapted | Weak |
| 5 | Capacity is not the limit | Fold 0 seed 46: encoder LR x5 train 8.30 to 7.26, validation 9.11 to 9.44; token MLP -0.013 (CI -0.051, 0.027) | LR is not capacity; no x2-3; never combined with augmentation; no mixup or drop path anywhere | Weak |
| 6 | Error is not data-limited | Learning curve dropped before running, because of "underfitting", a word later retracted | Never run | Never tested |
| 7 | Std pooling is settled | vs mean -0.0804 (CI -0.116, -0.048), folds 0, 4, seeds 46, 47 | Other statistics on fold 4 seed 46 only. Covariance, Gram and spatial correlation of tokens never tested | Strong vs mean; never tested beyond |
| 8 | Floors: HEX 8.6, CD 3.9-4.7 | 25 frames, one 2023 style, marker half-split | The repo also holds about 5.5 and 6.7; CV has no floor. It drove "no new exploration" on 10-31 | Weak |
| 9 | Localised pooling is closed | Seed 46: +0.071, +0.092; masked framed +0.031 | Replaced the whole frame rather than adding to it; not rerun with legible windows | Medium |
| 10 | Resolution and schedule are right | Schedule10, fold 4: -0.0399, -0.0985; canonical size +0.067, +0.235; zoom +0.088 | Fold 0 schedule confirmation has no record; no other stride | Weak |
| 11 | Detector is stuck at F1 0.86 | 25 frames: 0.862, 0.859, 0.865 | The other two detectors were never fed to hex6 or the graph. The detector with the lowest val-5 F1 gave the strongest hex6 gain, so F1 is a poor proxy | Medium for F1; weak as a limit |
| 12 | B is near the best mix; seeds are exhausted | Nested best -0.017; infinite seeds -0.024 from 2 seeds | Stored members only | Strong for stored members only |
| 13 | OOF mirrors test | OOF 8.877, test 8.9675, HEX +0.39 | One test sample; donor grouping holds for new-format IDs only (4514 old rows are singletons); no untouched fold for stretch or graph | Medium |
| 14 | Image values at graph nodes memorise | 32 dims +0.165, +0.144; 8 dims +0.041 | One token source; no regulariser | Weak |

## The greedy path

| Fork | Kept | Dropped, folds x seeds | Later change that could reverse it | Re-tested |
|---|---|---|---|---|
| Field of view | Crops, then whole frame | Whole frame first dropped at 9.18 | Recipe A | Yes, and reversed |
| Crop encoder, 10-01 | CPSAM 9.1338 | DINOv2 B/14 9.2125, ConvNeXt 9.4300; folds 2, 3 x seed 42 | SSL, std | No |
| Whole-frame encoder, 10-02 | CPDINO 9.302 | ConvNeXt 9.485, DINOv3 9.539, fold 3 x seed 42, no SSL, mean. CPSAM dropped on memory (98.86 GiB peak), one epoch | Std, SSL, schedule10 | Old recipe only |
| SSL | SSL5 | Three folds, one seed each; SSL20 on one fold | Std pooling | Seed only |
| LR and clip | 1e-3, clip 1 | Fold 3 seed 42, six arms | Std, schedule10 | Encoder x5 only |
| Pooling | Std 9.0479 | mean_std_min_max 9.0557, mean_std 9.0613, max 9.0941, mean 9.1333, min 9.1343; fold 4 x seed 46 | Other encoders | Only mean, four pairs |
| Hybrid | w = 0.5, Phase-1 members frozen | Per-metric -0.009 | Phase-1 members with std or schedule10 | No |

Std also interacts with the encoder: a std branch on CPSAM failed (+0.0015), then won on CPDINO. Std was never run on other encoders.

## Model families

| Family | Input | Budget against the winner |
|---|---|---|
| CPDINO ViT-B (winner) | Whole frame | SSL5, std, schedule10, four flips, 20 epochs, 5 folds x 2 seeds |
| CPSAM v2 ViT-L | 256 crops | 40 views, mean; SSL on one member |
| CPSAM 512 | Central crop | One epoch, fold 2: 9.4455 vs 256's 9.6909; follow-up has no record |
| DINOv2 B/14, ConvNeXt | Crops | Folds 2, 3, no SSL, mean |
| DINOv3 B/16, ConvNeXt | Whole frame | Folds 3, 0, one seed, no SSL, mean, old cosine |
| DINO B/8 | All-data SSL | Probe crashed on a metric bug; never scored |

Never tried: Swin, ResNet, EfficientNet, RETFound, ViT-L on the whole frame under recipe A, attention pooling on the whole frame, second-order pooling on tokens.

Missing records: `phase2_crossfold_20261015`, `phase2_backbone_ssl_20261020` and `phase2_pooling_20261020` are not in git, so only summaries exist.
