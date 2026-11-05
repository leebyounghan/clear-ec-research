# Submission-2 package: hybrid B + hex6 HEX correction (2026-11-01)

For the second Phase II submission only. Submission 1 stays `results/submission1_package_20261031/`.

| File | Size | SHA256 |
|---|---:|---|
| `clear-ec-hybrid-hex6_20261101.tar.gz` (docker save, gzip -6; image `clear-ec-hybrid-hex6:20261101`, 11.3 GB unpacked) | 7.24 GB | `a29f568cd0aab67ae2d3fa00e4c84681817998c92fb86416b508d6d7bb00ab52` |
| `model_hybrid_20261030.tar.gz` (identical to submission 1, same hash; no re-upload needed) | 9.21 GB | `1134b453653b4a7215532be0be9dddc5f2a29d66af02cdaf4a0f7dbae506bf87` |

Build box: `~/workspace/clear-ec/package_20261101/` (model archive is a hard link to the submission-1 copy).
PI's Mac: `~/Downloads/clear-ec-upload_20261101/` (image only, downloaded 2026-11-01 with eight ranged ssh streams and
checked against the hash above). Build context: `submission_hybrid_hex6_20261101/`.

## What changed relative to submission 1
The ten hex6 detectors (nine fine-tuned families + pre-only s46, 3.4 GB) are baked into the image at `/opt/app/hex6`,
so the model archive does not change. After the guarded hybrid prediction, `hex6_t4_20261101.py` (fp16 autocast,
memory-efficient/math SDPA, because the T4 has no bf16 and no flash attention) computes the mean log six-neighbour
fraction over the ten detectors and corrects HEX: `exp((1-w) log HEX + w (a + b f))`, a -0.2084, b 0.5989, w 0.28.
CD and CV are untouched. If hex6 raises, the hybrid HEX is kept and the error is logged (`errors.hex6`). The
constant fallback (unreadable input) is never corrected.

Expected effect: HEX LOFO on hybrid B -0.031 pp mean (all rows), -0.021 on black-frame images, -0.040 on unframed;
fp16 re-computation -0.0326 (`results/green_track_20261028/t4_numerics/result.json`).

## Verification (`results/submission2_test_20261101/sub2_container_test_20261101.log`, RTX 4090)
- CD and CV byte-identical to submission 1 on all four normal cases; HEX before correction equals submission-1 HEX.
- hex6 applied on all normal cases; HEX change -0.8 .. +3.0 %.
- Faults, all exit 0 with three valid outputs: RGB and uint16 -> converted, corrected; all-zero frame -> no peaks,
  HEX kept; 960x1280 crop -> phase1_only, hex6 skipped; unreadable -> constant; injected hex6 fault -> output equal
  to submission 1 byte for byte; injected recipe A fault -> phase1_only, corrected.
- Wall time 35-42 s per case on the 4090 (submission 1: 22 s). T4 time is the open risk: estimate 130-150 s per case
  against the 300 s limit; Try-out on the T4 decides. Fallbacks if it is too slow: A10G instance, then the
  frozen-head hex6 family (-0.011, no extra ViT passes).

## Expected Try-out outputs (4090 fp16; T4 may differ slightly, mostly in HEX)
| Case | CD | CV | HEX |
|---|---:|---:|---:|
| 0001-19 | 2873.6004 | 0.390776 | 0.535880 |
| 0001-20 | 2027.3348 | 0.466158 | 0.434209 |
| 0001-21 | 4150.6763 | 0.319676 | 0.617164 |

## Platform settings
Same as submission 1: GPU NVIDIA T4, memory 32 GB. Upload only the new image; keep the existing model archive active.
