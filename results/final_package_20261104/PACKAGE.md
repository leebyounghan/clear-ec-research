# Final-candidate package: hybrid B + hex6 + stretch (2026-11-04)

Not submitted. This is a ready part for the last Phase II submission, planned for 2026-11-12 or 11-13. It is rebuilt only if a better
base model passes its gates first. Image `clear-ec-hybrid-final:20261104`, image id `sha256:2a55a25ad741…`, 11.3 GB unpacked.
Model archive unchanged from submission 1, `model_hybrid_20261030.tar.gz` (sha256 `1134b453…`), so only the image needs uploading.

- Build context: `submission_final_stretch_20261104/` (the hex6 weights live only on the build box, `~/workspace/clear-ec/`).
- Archive on the build box: `~/workspace/clear-ec/package_20261104/clear-ec-hybrid-final_20261104.tar.gz`, hash in `SHA256SUMS.txt`.
  Not yet copied to the PI's Mac.

## What changed relative to the 2026-11-01 hex6 image
One stage after hex6: per metric, `p' = c exp(g (log p - mu) + mu)` (`final_stretch_20261104.py`). If hex6 was not applied to the image,
HEX uses the B-only slope. Skipped on the constant fallback. A stretch failure keeps the value before it and is logged.

| Metric | g | c | mu | LOFO g |
|---|---:|---:|---:|---|
| CD | 1.075 | 0.99963 | 7.91639 | 1.075 on four folds, 1.05 on fold 4 |
| CV | 1.025 | 0.99993 | -1.00075 | 1.025 on all folds |
| HEX, hex6 applied | 1.175 | 0.99793 | -0.62425 | 1.175 on four folds, 1.2 on fold 4 |
| HEX, hex6 not applied | 1.075 | 1.00006 | -0.62664 | not repeated |

## Expected effect (`results/meta_ceiling_20261103/stretch_final.json`)
| Comparison | Delta mean | 95 % CI |
|---|---:|---|
| LOFO hex6 + stretch vs B, all rows | -0.070 | -0.087, -0.054 |
| Slope -0.025 / +0.025, full-data fit | -0.073 / -0.071 | |
| Deployed composition (Phase-1 six, folds 2-4) vs that B | -0.074 | -0.096, -0.052 |
| Of which stretch alone, on that composition | -0.046 | -0.064, -0.027 |

Caveats: found while looking at all five folds, so no untouched confirmation fold exists. The OOF has two recipe A seeds; the image has
three, on full-data models. Spread for Phase-1 six vs five differs by under 0.2 %; for three full-data seeds it is unmeasured.

## Verification (`final_container_test_20261104.log`, RTX 4090): all checks passed
- All four normal cases equal the hex6 image's stored output with the stretch applied (relative tolerance 1e-5). Mode hybrid, no errors, ~34 s per case.
- Faults, all exit 0 with three valid outputs: RGB, uint16, all-zero frame, 960x1280 crop, unreadable (constants, no stretch), and injected hex6, recipe A and stretch faults.
- The injected stretch fault reproduces the hex6 image output.

## Expected Try-out outputs (4090 fp16; T4 may differ slightly)
| Case | CD | CV | HEX |
|---|---:|---:|---:|
| 0001-19 | 2882.6722 | 0.391346 | 0.534808 |
| 0001-20 | 1981.2159 | 0.468901 | 0.417676 |
| 0001-21 | 4280.2061 | 0.318539 | 0.631343 |

## Open before any submission
Runtime on the T4 is unverified for any hex6 image (estimate 130-150 s per case against the 300 s limit). A platform Try-out decides, and that is a PI account action. Fallbacks if too slow: A10G instance, then a frozen-head hex6 family (-0.011).
