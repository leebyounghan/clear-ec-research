# September 11 — native, fold-clean SSL screen

Fresh controls were trained after exact-duplicate-linked slide groups were
removed from training. The same 3,568 primary validation rows were used for
the comparison.

| Predictor | Standalone mean | Fixed-six mean |
|---|---:|---:|
| Clean 35-epoch control | 9.261761 | 8.975492 |
| Custom masked-EMA SSL + regression | 9.208103 | 8.971051 |
| Continued-supervised 55-epoch control | 9.265722 | — |

Both folds improved for the SSL replacement, but the 0.004441 ensemble gain
missed the predeclared 0.02 gate. No immediate promotion followed.
Training-only feature diagnostics are not evidence of accuracy.

Recorded training/pretraining/evaluation totaled about 27.36 GPU-hours;
queue allocation was about 28.86 GPU-hours, not billing.
Later raw-label comparisons are summarized [here](RAW_SSL_NEXT_20260913.md).

[Detailed historical record](../archive/snapshot_20260915/docs/SSL_SCREEN_20260911.md) — retained for provenance, not reviewer setup.
