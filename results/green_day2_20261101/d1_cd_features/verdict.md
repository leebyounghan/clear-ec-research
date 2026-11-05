# Green day 2 — D1 / D1b / D2 verdicts (2026-11-01, 20:30-20:45 UTC)

Base = hybrid B (8.8772 on 8954 rows; 8.8682 on the 8934 non-leaked rows used here: CD 6.307, CV 10.186, HEX 10.111).
Detectors = the three pre-trained-only (srep) stride-2 detectors, seeds 46/47/48, fold-clean OOF peaks; they are near
full-frame (about 660 peaks per image, bounding box 72% of the frame), unlike the fine-tuned detectors (compact clusters).

## D1 — CD from full-frame detections: CLOSED (w = 0 everywhere)

| estimator (family mean of 3 seeds, threshold 0.35) | est. CD MAPE | r(err, B err) | LOFO w (5 folds) | delta |
|---|---|---|---|---|
| nn_cd: hexagonal-lattice density from median NN spacing of interior peaks | 9.42 | 0.69 | 0 | 0.0000 |
| vor_cd: 1 / median bounded Voronoi cell area (interior, <= 3x median) | 11.37 | 0.60 | 0 | 0.0000 |
| vor_cd_hex: same, 6-sided cells only | 11.36 | 0.60 | 0 | 0.0000 |
| win_dense / win_cluster / win_typical / win_quality (538x408 windows) | 12.5-13.2 | 0.56-0.58 | 0 | 0.0000 |
| dens_tissue: peaks per non-black area | 13.30 | 0.57 | 0 | 0.0000 |

Stratified (framed / unframed, w chosen per stratum): also 0 everywhere. Per stratum the picture is the same on both sides:
nn_cd 9.0 vs B 5.3 (unframed), 9.7 vs 7.0 (framed), error correlation 0.73-0.74, above B/estimator error ratio -> no blend can help.
Threshold 0.5 variants: identical conclusion.

**Oracle.** The expert-point CD (`CD_from_hull`, 148 cells in a 0.05 mm^2 hull) on the 25 annotated images is 7.24% MAPE
against the device CD (LOO log-linear calibrated 7.51%); B is 7.66% on the same 25. So even a perfect detector counting
exactly the expert's cells would not beat the regressor on average; detector-based CD is structurally capped near 7%.

## D1b — quality-gated blending: CLOSED (w = 0 in every stratum)

Blend weight chosen per tercile/quintile of label-free quality covariates (inter-seed agreement of log nn_cd, interior
cell count, window quality score, framed x agreement): w = 0 in every stratum for nn_cd, vor_cd and vor_cd_hex.
B is better than nn_cd even in the best-agreement tercile (CD MAPE 5.0 vs 7.9; worst tercile 7.8 vs 12.1).
The 25 annotated images (nn_cd 4.6% vs B 7.7%) are a small-sample selection effect, not a usable regime.

## D2 — hex6 variants for framed images: gate failed; coverage gating promising

| HEX feature (3 pre-only seeds) | plain LOFO mean pp | stratified | framed HEX pp | unframed HEX pp |
|---|---|---|---|---|
| hex6_all (control) | -0.0138 (5/5) | -0.0171 | -0.013 | -0.087 |
| hex6_interior (cells >= 15 px from black/border) | -0.0137 | -0.0170 | -0.013 | -0.086 |
| hex6 in the cluster window (538x408 at the compact detector's centroid) | -0.0141 | -0.0147 | -0.028 | -0.059 |

The tissue-interior rule changes almost nothing (bounded cells are already away from the border). The window restriction
doubles the framed-image gain but loses more on unframed images, so it fails the overall half of the gate. The compact
detector's window overlaps the expert ROI box with median IoU 0.81 (20/25 leaked; indicative).

Coverage-conditional weight (D1b, HEX part): w per tercile of interior cell count = 0.08 / 0.18 / 0.40 lifts the
3-seed hex6 from -0.0138 to -0.0174 mean pp (+0.0036, below the 0.005 gate alone). Tested inside the active ten-family
composition as **D2b** (separate verdict).

## Consequence for the GPU plan

The CD route needs no more detector work (ceiling, not recall). GPU time goes to HEX: TTA / frame-augmented pre-only
detectors (G1/G2, evaluated as hex6 members) and the hex6-auxiliary multi-task fine-tune of the regressor (G3).

Files: `d1_cd_features/{features.npz,result.json,evaluate.log}`, `d1b_quality_gate/result.json`, `source/green_day2_cd_features_20261101.py`, `source/green_day2_d1b_quality_gate_20261101.py`.
