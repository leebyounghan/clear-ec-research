# September 14 release — sixth-only SSL + raw labels

The first five checkpoints, six-member equal weighting, calibration and original
Docker image are retained. Only `full_quality_seed77.pt` is replaced.
See [method](METHODOLOGY.md), [artifact hashes](ARTIFACTS.md) and
[Docker execution](../../docker_review_20260914/README.md).

## Fixed recipe

| Stage | Configuration |
|---|---|
| Initialization | Original `cpsam_v2` |
| SSL data | All 9,000 challenge training images; labels unused; no external corpus |
| SSL objective | Same-crop masked EMA feature prediction + variance 0.1 / covariance 0.01 |
| SSL optimization | 6,000 updates, batch 32, AdamW LR 1e-5, weight decay 0.05, warmup/cosine |
| View policy | Native 256 full-quality tiles, D8, mild intensity changes; no resize |
| Regression initialization | Terminal EMA encoder with fresh heads |
| Regression data | 8,998 valid original-label rows; no slide averaging |
| Regression fit | Seed 77, two crops, batch 48, mixed loss, 5,215 fixed updates |
| Regression LR | Encoder 1e-5; head/norm 1e-3 |

Implementation: [ssl_native_core.py](../../code/ssl_native_core.py).
For a new fit without past research-run folders, use the
[standalone training adapter](../../training_reproduction/README.md).
No independent holdout score exists for this all-data fit. The
[matched development comparison](RAW_SSL_NEXT_20260913.md) is separate from
the verified official score of 8.7010.

## Completion evidence

Times below are UTC; the platform's September 14 date is a different calendar context.

| Event | Recorded completion |
|---|---|
| SSL | September 14, 20:34; 7,221.98 seconds; all 9,000 IDs sampled |
| Regression | September 15, 00:51; 15,402.14 seconds |
| Archive and member hash verification | September 15, 01:02 |
| Local full-archive SHA256 verification | September 15, 02:09 |

SSL plus regression consumed 6.284 single-H200 process-hours, excluding
preparation, runtime checks, compression and transfer.

Strict reload passed. The eight-image frozen CUDA 12.6 application check found
unchanged first-five predictions and maximum FP16/BF16 relative drift of about
0.0651%. This is not an actual archived-Docker/T4 benchmark.
The other seven archive resources (first five, foundation and configuration)
matched their preserved hashes.

Model archive: `model_full_quality77_ssl_raw_20260914.tar.gz`,
7,964,665,157 bytes. The official 8.7010 evaluation is attached to model `b4137c8d`;
its platform CRC64-NVME exactly matches the Drive archive. See the
[verified platform binding](ARTIFACTS.md#official-evaluation-and-shared-files).
Detailed execution/transfer logs are archived, not current instructions.

[Detailed historical record](../archive/snapshot_20260915/docs/SSL_RELEASE_20260914.md) — retained for provenance, not reviewer setup.
