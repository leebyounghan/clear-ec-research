# E7 — hex6 detection-threshold operating point (2026-11-03, 17:22 UTC): FAIL; the shipped 0.35 is the best of five

Pre-registered rule: pick the threshold whose validation-5 (miss, FP) gives the lowest E4-simulated oracle HEX MAPE,
then one LOFO run of the active ten on B.

| threshold | val-5 miss | val-5 FP | simulated HEX MAPE | active ten LOFO on B (mean pp) | HEX pp | w (fold 0-4) |
|---|---|---|---|---|---|---|
| 0.20 (pre-registered pick) | 0.193 | 0.261 | 17.24 | -0.0253 | -0.0759 | 0.22-0.28 |
| 0.25 | 0.201 | 0.252 | 17.27 | -0.0263 | -0.0790 | 0.24-0.30 |
| 0.30 | 0.209 | 0.239 | 17.27 | -0.0285 | -0.0856 | 0.24-0.32 |
| **0.35 (shipped)** | 0.221 | 0.225 | 17.34 | **-0.0311** | -0.0934 | 0.26-0.32 |
| 0.50 | 0.279 | 0.155 | 17.87 | -0.0194 | -0.0581 | 0.20-0.24 |

The threshold barely moves the validation-5 operating point (miss 0.19-0.28, FP 0.16-0.26) and the simulation is
flat (17.2-17.9), but on the 8954 rows the LOFO gain falls monotonically on both sides of 0.35: +0.0058 worse at 0.20,
+0.0117 worse at 0.50. Reading: the E4 simulation models FPs as random boundary / near-cell points and misses as random,
whereas the real low-threshold detections are noise in poor-quality regions (correlated, not random) and the real
misses at high threshold are the low-contrast cells whose absence breaks neighbourhoods. The simulation's "FPs are
cheap for HEX" does not transfer; the shipped 0.35 stays. Gates fail (gain in the wrong direction).

Files: `e7_threshold/{result.json, run.log}`, `source/green_day3_e7_threshold_20261103.py`.
