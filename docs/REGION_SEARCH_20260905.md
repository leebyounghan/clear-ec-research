# September 5 — native region selection

The screen compared full_uniform, full_quality and anchor_quality policies
using only native 256-pixel crops from original images. Models used two training
crops and five locations × eight transforms at evaluation; no resized global
view or ROI ground truth was required.

Quality selection ranks contrast, Laplacian sharpness and unsaturated fraction,
then enforces spatial separation. It can still select unrepresentative regions
and is not validated ROI recovery.

The full_quality policy became the sixth release member after confirmation and
integration. External-only pretraining was a separate research branch, not the
additional SSL stage used in the September 14 release.

[Detailed historical record](archive/snapshot_20260915/docs/REGION_SEARCH_20260905.md) — retained for provenance, not reviewer setup.
