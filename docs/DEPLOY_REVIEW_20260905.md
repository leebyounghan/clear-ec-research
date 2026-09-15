# September 5 — initialization and inference audit

Source/initialization consistency and submission inference were checked after
the earlier training defects were identified. Historical and regional builders
consume RNG differently; their frozen sources must not be silently interchanged.

Verification of checkpoint loading, output interfaces and numerical consistency
is distinct from a new held-out accuracy measurement.
See [reproduction scope](REPRODUCE.md) for what has and has not been verified.

[Detailed historical record](archive/snapshot_20260915/docs/DEPLOY_REVIEW_20260905.md) — retained for provenance, not reviewer setup.
