# Reproduction scope

## Inference from the released weights

Use the [Docker guide](../docker_review_20260914/README.md).
The archived image and separately mounted September 14 model reproduce the
specified inference pipeline without our Kubernetes environment or host venv.

| Check | Evidence |
|---|---|
| Archive and extracted-resource identity | SHA256 verification passed |
| Checkpoint compatibility | Strict reload passed |
| Frozen CUDA 12.6 application | Eight-image check passed; first-five outputs identical |
| FP16/BF16 comparison | Maximum relative drift approximately 0.0651% |
| Fresh GitHub checkout + archived Docker on RTX 4090 | One-image inference passed on September 15; [verification record](DOCKER_SMOKE_20260915.md) |
| Anonymous Drive downloads | Both complete archives downloaded and hashes verified on a separate Linux host |
| Actual archived Docker on T4 | Not independently verified by this documentation update |

Hash equality does not establish predictive accuracy or new-device numerical parity.
Historical H200 timings are not T4 benchmarks.

## Training from source

[Historical training recipes](../review_submission_20260910/TRAINING.md) document
the first-five all-data fits and the earlier regional model.
The replacement sixth uses the [September 14 SSL/raw recipe](SSL_RELEASE_20260914.md).
The [technical report](../output/pdf/Lamda_KIM_CLEAR_EC_Technical_Report.pdf)
describes data handling, schedules and model selection.

Full clean retraining has not been certified. Remaining gaps include validation
of the authorized raw-data bootstrap, the historical training-environment lock,
and the complete selection/calibration dependency chain. Do not substitute the
inference dependency lock for a bitwise training guarantee.

Historical folds were reused for development; later exact-duplicate cleanup
does not make them an untouched test set. Slide grouping is not verified patient
grouping. Raw images and labels must come from the authorized challenge source,
not this repository.

[Detailed historical record](archive/snapshot_20260915/docs/REPRODUCE.md) — retained for provenance, not reviewer setup.
