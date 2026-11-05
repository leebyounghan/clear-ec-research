# Submission-1 package, guarded rebuild (2026-10-31)

Supersedes `results/submission1_package_20261030/` for upload. Same model archive; only the container changed.

| File | Size | SHA256 |
|---|---:|---|
| `clear-ec-hybrid_20261031.tar.gz` (docker save, gzip; image `clear-ec-hybrid:20261031`) | 4.05 GB | `31c5f3bea337fb2e1f11cf710039bba5784157dc6bef9470e0b02a4bc216a962` |
| `model_hybrid_20261030.tar.gz` (unchanged; extracts to `/opt/ml/model`) | 9.21 GB | `1134b453653b4a7215532be0be9dddc5f2a29d66af02cdaf4a0f7dbae506bf87` |

Build box: `~/workspace/clear-ec/package_20261031/`. Upload copies on the PI's Mac: `~/Downloads/clear-ec-upload_20261031/`.
Build context: `submission_hybrid_20261030/` (Dockerfile now also copies `hybrid_guard_20261031.py`).

## What changed and why
`hybrid_guard_20261031.py` isolates per-image failures, because a crash on a hidden test image cannot be seen or
reproduced by us and would fail the whole evaluation. Recipe A fails -> Phase-1 alone; Phase-1 fails -> recipe A
alone; both fail or the MHA is unreadable -> MAPE-optimal training constants (CD 2770, CV 0.36, HEX 0.53). Non-uint8
or multi-channel input is converted to 2-D uint8 first (identity for the validated case). Start-up checks (CUDA,
memory-efficient SDPA for 20,336 tokens, model files) stay loud so a broken runtime shows up in Try-out instead of
silently degrading every case. The job log line carries `mode`, `errors` and `notes`.

## Verification (`guard_test_20261031.log`, RTX 4090, `--network none --memory 16g`)
- Equivalence: the guarded image reproduces the verified 2026-10-30 image byte for byte on 4/4 normal cases
  (the three platform debug MHAs 0001-19/20/21 exactly as Grand Challenge served them, and 0234-20).
- Platform cross-check: the Phase-1 half on the 4090 matches the Phase-1 outputs the platform produced on its T4
  for the same three MHAs (2026-09-14 jobs) to a relative 1.5e-6.
- Faults, all exit 0 with three valid outputs: RGB input -> hybrid (converted); uint16 -> hybrid (converted);
  all-zero frame -> hybrid; 960x1280 crop -> phase1_only; unreadable file -> constant; injected recipe A /
  Phase-1 / both faults -> phase1_only / recipea_only / constant.
- Wall time per case on the 4090: 22 s (Phase-1 13.6 s, recipe A 7.4 s, rest start-up). Phase-1 alone measured
  44 s exec on the platform T4; the hybrid is expected near 80-120 s on T4, to be confirmed by Try-out (limit 300 s).

## Expected Try-out outputs (fp16; T4 may differ in the 6th significant digit)
| Case | CD | CV | HEX |
|---|---:|---:|---:|
| 0001-19 | 2873.6004 | 0.390776 | 0.520125 |
| 0001-20 | 2027.3348 | 0.466158 | 0.437548 |
| 0001-21 | 4150.6763 | 0.319676 | 0.618357 |

## Platform settings
Image: GPU NVIDIA T4, memory 32 GB (Phase II allows <= 32 GB; the container passed with a 16 GB cap, 32 removes
the instance-overhead margin risk). Model archive is 15 % larger than the 8.0 GB Phase-1 model the platform
accepted on 2026-09-14; the upload page states no size limit.
