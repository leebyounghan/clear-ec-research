# sg2_specialist — verdict (2026-10-28, seed 46; TTA and seeds 47/48 pending)

Confirmation on an independent detector. The stride-2 pixel-shuffle head trains in 6 min per fold
(val-5 F1 0.62-0.67 vs 0.63-0.66 for the stride-8 specialist; fold 4 0.666 vs 0.642) and yields 210-300 peaks
per frame. HEX<-hex6_all (p2b variant F, pre-registered): -0.063 pp HEX (-0.021 pp mean), 5/5 folds
(-0.105/-0.061/-0.075/-0.041/-0.032), w 0.18-0.20, calibrated hex6 MAPE 11.3%, error correlation 0.85.
Below the gate (-0.09 HEX). Spacing-based CD is worse at stride 2 (11.6% vs 8.9%): more peaks but noisier
spacing; no CD or CV gain. Next: four-flip TTA of the heatmap and seeds 47/48 (queued) to test whether the
remaining gap is detection noise; hexagon-shape features (p4) and detector combination (p5) on CPU.

## Addendum 19:45 UTC — four-flip TTA of the heatmap

No denoising: HEX<-hex6_all -0.057 pp (5/5 folds, w 0.18-0.20, r_err 0.85) vs -0.063 single-view; the peak
set changes (195-240 peaks per image vs 210-300) but the HEX information does not. Spacing CD improves
(11.6 -> 9.8% MAPE) yet blends at w = 0. View averaging is not where hex6's noise comes from; the seed
ensemble (p6) is the remaining denoising test.
