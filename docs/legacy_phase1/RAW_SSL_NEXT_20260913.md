# September 13 — raw-label and SSL comparison

Six regression fits across folds 2/3/4 completed. All rows below use the same
5,350 primary validation images; lower uncalibrated mean MAPE is better.

| Sixth-member recipe | Standalone | Fixed six |
|---|---:|---:|
| Original CPSAM + raw labels | 9.271957 | 9.029185 |
| Standard SSL + raw, encoder LR 1e-5 | 9.222232 | 9.019599 |
| Standard SSL + raw, encoder LR 5e-6 | 9.229801 | 9.021819 |

Standard-rate SSL improved the fixed ensemble in all three folds; aggregate
gain was 0.009586. Half LR did not beat standard overall. The raw-only control
is not the September 10 slide-smoothed baseline.

Each fold excluded validation images and duplicate-linked training slide groups
from SSL and regression. These are reused development folds, not official scores.

SSL consumed 6.010 GPU-hours, regression 25.845 and evaluation 1.221:
33.077 measured GPU-hours in total (rounding applies).
[Aggregate audit](../../results/raw_ssl_next_20260913/ssl_audit.json).

A raw-only all-data package was prepared separately. The standard-rate SSL
result subsequently informed the [September 14 release](SSL_RELEASE_20260914.md);
a development checkpoint is not the all-data release checkpoint.

[Detailed historical record](../archive/snapshot_20260915/docs/RAW_SSL_NEXT_20260913.md) — retained for provenance, not reviewer setup.
