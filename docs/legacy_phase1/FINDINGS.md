# Development findings

The release combines direct regression, native multi-view inference, complementary
training recipes and full-field quality-selected crops. Earlier segmentation,
ROI, spectral and feature-fusion trials did not establish a stronger submitted
candidate in their tested settings.

Historical claims of a proven CV/HEX noise floor were too strong. A small
25-image ROI sample, shared preprocessing and correlated errors do not establish
irreducibility. Optimizer and label-handling defects were subsequently corrected;
historical scores must be read with their cohort and validation caveats.

The matched SSL/raw result used for the September 14 release is summarized in
[RAW_SSL_NEXT_20260913.md](RAW_SSL_NEXT_20260913.md).

[Detailed historical record](../archive/snapshot_20260915/docs/FINDINGS.md) — retained for provenance, not reviewer setup.
