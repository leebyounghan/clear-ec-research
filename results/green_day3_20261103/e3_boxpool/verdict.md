# E3 — recipe A with std pooling restricted to the detector cluster window (2026-11-03, research): FAIL, worse than standard on both folds

Seed 46, 20 epochs, same recipe and SSL init as the standard member; pooling 'std_box' = std over the patch tokens inside
the 538x408 window centred on the compact stride-2 detector cluster (D1 window; IoU 0.81 with the expert ROI on the 25,
0.71 on the clean five), carried as a second input channel and split off before the encoder; rows without a window use
the full frame (109). Fold 0 on GPUs 0-1 (15:38-19:59 UTC), fold 1 on GPUs 2-3 (16:43-21:06 UTC). Single-view terminal-
epoch predictions compared with the standard member's single view (TTA for the box run queued, not decisive).

| fold | standard (CD / CV / HEX / mean) | box (CD / CV / HEX / mean) | delta mean [95 % CI, donor bootstrap] | framed / unframed delta | masked-pool probe |
|---|---|---|---|---|---|
| 0 | 6.348 / 10.455 / 10.538 / **9.114** | 6.465 / 10.495 / 10.596 / **9.185** | +0.071 [-0.005, +0.151] | +0.070 / +0.073 | 9.122 (+0.008) |
| 1 | 6.497 / 10.192 / 10.037 / **8.909** | 6.700 / 10.287 / 10.017 / **9.001** | +0.092 [+0.022, +0.164] | +0.083 / +0.101 | n/a |

TTA4 (box run's own four-flip pass, 21:18 UTC): fold 1 standard 8.881 vs box 8.976, delta +0.096 [+0.031, +0.164]
(CD +0.20, CV +0.12, HEX -0.04; framed +0.054, unframed +0.133). Fold 0 stays single-view in the table because the
masked-pool probe in the same comparison has no TTA file; the box-vs-standard TTA gap there is the same sign.

Per target the box model loses most on CD (+0.12 / +0.20) and CV, with HEX flat (+0.06 / -0.02). Training curves:
fold 0 best epoch 16 (9.105), terminal 9.1x; fold 1 best epochs 16-17 (8.94), terminal 9.00 — the restricted statistic
is not better even at its best epoch.

Reading: restricting the std statistic to ~17 % of the tokens discards context the regressor uses (the window is where
the technician measured, but the whole-frame token statistics carry the image-quality and density information that
the CD head relies on), and the window itself is only a 0.7-0.8 IoU proxy. Together with roi_crop (+0.002), local_pool
(-0.016 vs its own global control) and the masked-pool probe (+0.008), every attempt to localise recipe A's pooling to
the measured region has been null or negative. Closed (research gate: >= 0.05 better on both folds).

Files: `e3_boxpool/{compare_fold0.json, compare_fold1.json, cluster_boxes.npz, recipeA_box_pool_backbone_20261103.py, f0_s46_stdbox/, f1_s46_stdbox/}`, `source/green_day3_e3_compare_20261103.py`.
