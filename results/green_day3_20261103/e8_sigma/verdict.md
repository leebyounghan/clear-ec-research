# E8 — localisation-aware stride-2 detectors (target sigma 5 / 3.5 px vs 8) (2026-11-03, 21:35 UTC): FAIL on both gates

Same recipe as the shipped stride-2 specialist (fold-clean init, 1000 updates, batch 2, four flips, 20 train-green
images), only the Gaussian target width changed. Five folds each (sigma 5 on GPUs 0-1 19:59-20:27; sigma 3.5 on GPUs 2-3
21:06-21:33).

| detector | val-5 miss | val-5 FP | val-5 jitter (x spacing) | val-5 F1 | hex6 alone on B (mean pp) | swap for sg2_s46 in the ten | ten + it |
|---|---|---|---|---|---|---|---|
| sigma 8 (shipped sg2_s46) | 0.293 | 0.188 | 0.195 | 0.644 | -0.0086 | (control -0.0311) | |
| sigma 5 | 0.264 | 0.094 | 0.179 | 0.646 | -0.0062 | -0.0303 | -0.0312 |
| sigma 3.5 | 0.424 | 0.049 | **0.162** | 0.604 | -0.0009 | -0.0307 | -0.0310 |

Localisation gate (jitter <= 0.17 with miss not worse than +0.02 vs 0.221 at threshold 0.35 on the val-5 operating
point): sigma 5 misses it on jitter (0.179), sigma 3.5 passes jitter but misses 42 % of the cells. Sharper targets
trade recall for precision and localisation along one curve; E4 needs both ends at once (misses <= 8 %, jitter <= 0.17),
which no point on this curve reaches. HEX gates: nothing above +0.0001 over the control. Closed.

Files: `e8_sigma/{result.json, evaluate.log, evaluate_sigma5_early.log}`, detectors under `experiments/green_track_20261028/sg2_sigma5/`, `sg2_sigma3p5/`, `source/green_day3_e8_sigma_20261103.py`.
