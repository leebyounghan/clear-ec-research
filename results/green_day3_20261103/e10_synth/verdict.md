# E10 — route B: detector trained on synthetic endothelium only (2026-11-03 23:35 to 2026-11-04 05:16 UTC): FAIL, route B closed

Protocol `e10_synth/protocol.json`. Renderer loop (parent) v1 -> v5, measured each pass with the shipped pre-only s46 on
24 synthetic images (CPU): v1 miss 0.087 / FP 0.002 / jitter 0.061 / F1 0.952; v2t4 0.368 / 0.021 / 0.088 / 0.753;
v3t1 0.282 / 0.017 / 0.098 / 0.829; v4t1 0.303 / 0.184 / 0.109 / 0.759; v5t1 0.453 / 0.094 / 0.150 / 0.672; final v5
(waviness capped) 0.321 / 0.119 / 0.127 / 0.762, against real validation-5 0.166 / 0.166 / 0.222 / ~0.64.

## Step 2 — synthetic-only detector on the 25 real annotated images (all clean), threshold 0.35, hull region, 0.5-spacing match

| detector | subset | miss | FP | jitter (x spacing) | F1 at 12 px |
|---|---|---|---|---|---|
| synthetic-only (v5, 3000 images, 4000 updates, fold-4 encoder init, loss 0.61) | all-25 | 0.182 | 0.542 | 0.309 | 0.744 |
| synthetic-only | validation-5 | 0.241 | 0.618 | 0.325 | 0.706 |
| shipped pre-only s46 (same code) | all-25 | 0.150 | 0.165 | 0.215 | 0.862 |
| shipped pre-only s46 | validation-5 | 0.170 | 0.192 | 0.214 | 0.857 |
| E4 break-evens | | 0.08 | 0.14 | 0.17 | |

Gate (all-25 miss <= 0.15 or jitter <= 0.17 with FP <= 0.30): fails on every term. The synthetic-trained detector is worse
than the shipped one on all four axes: 20 % more misses, 3.3x the false positives, 1.4x the localisation noise. The
renderer loop brought the synthetic proxy numbers of the *shipped* detector close to real (misses, FP), but a detector
trained on that appearance still fires on real texture it never saw as background (FP 0.54) and places centres less
precisely; the final training loss (0.61 vs ~0.2-0.3 for real-trained heads) says the v5 targets are also hard to fit.
Step 3 (fine-tune on train-20 + hex6 member check) is not run: nothing moved toward the E4 break-evens.

Route B (synthetic data as the no-label path to detector accuracy) is closed. Human point annotation is ruled out by the
PI (the challenge goal is a bridge without human help), so the open question is pushing the shipped pre-only detector,
which already uses no challenge labels, further without new human labels (E11).

Note on references: the "~0.64" F1 quoted for real validation-5 in earlier messages came from the fine-tuned
specialists' own validation F1 (a different matching region); by this code the shipped pre-only reaches 12-px F1 0.857
on validation-5 and 0.862 on all-25. The final v5 proxy (miss 0.321 / FP 0.119 / jitter 0.127 / F1 0.762) was therefore
harder than real on misses, not matched, and easier on jitter.

Files: `e10_synth/{protocol.json, score25.json, train.json, detector_synth.pt, domain_check_v5.json, domain_check_cpu_v1..v5t1.json}`,
`source/green_day3_e10_synth_20261103.py`, `run_e10_lane01.sh`.
