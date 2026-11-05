# September 6-7 — region confirmation and integration

Regional seed confirmation was followed by six-model inference integration.
Tests covered strict configuration checks, native views, unchanged first-five
predictions, decoded averaging, calibration, and the three-JSON output interface.

For full_quality, historical five-fold precision-audit means were 9.016114
(BF16) and 9.016264 (FP16); the exploratory cross-fitted calibration estimate
was 8.987422. These are reused development measurements, not official scores
or unbiased generalization estimates.

The September 10 package selected full_quality. The September 14 package
retains that inference policy and replaces only its sixth checkpoint.

[Detailed historical record](../archive/snapshot_20260915/docs/REGION_SEEDS_20260906.md) — retained for provenance, not reviewer setup.
