# September 12 — target policy replication

Folds 2/3 and a separate fold 4 compare raw versus slide-averaged labels:
fresh CPSAM initialization, seed 77, native full_quality two-crop training,
batch 48, mixed loss, and fixed 40-view evaluation. No extra SSL is used in
these target-policy controls.

Fold 2/3 use 5,215 regression updates; fold 4 uses 5,180.
Slide and exact-duplicate-connected overlap was removed. Primary validation
counts are 1,781 / 1,791 / 1,778, totaling 5,350.

These historically reused folds are not an untouched test set. Compare against
fresh matched smoothed controls, not an unmatched historical sixth member.
The later [raw/SSL audit](RAW_SSL_NEXT_20260913.md) uses the same primary cohort.

[Detailed historical record](archive/snapshot_20260915/docs/TARGET_REPLICATION_20260912.md) — retained for provenance, not reviewer setup.
