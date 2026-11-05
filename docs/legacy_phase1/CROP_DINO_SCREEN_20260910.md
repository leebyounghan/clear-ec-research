# September 10-11 — artifact-guard crop screen

Four fits completed. On the same 3,582 reused validation images:

| Predictor | Mean MAPE |
|---|---:|
| Current six-member control | 8.975950 |
| Fixed add7, matched original-crop control | 8.982686 |
| Fixed add7, artifact_guard_v1 | 8.964352 |

The guarded crop improved the six-member control by 0.011599 but missed the
predeclared 0.02-point gate. No promotion followed.
This positive but insufficient screen does not establish general ROI recovery.
The September 14 release retains its existing full_quality inference policy.

[Detailed historical record](../archive/snapshot_20260915/docs/CROP_DINO_SCREEN_20260910.md) — retained for provenance, not reviewer setup.
