# style_specialist_20261102 — verdict (2026-11-03 01:50 UTC): REJECTED, stop rule fired

Framed-style specialist (recipe A trained only on black_frac > 0.01 rows, about 3,450 per fold) vs the generalist
seed-46 member with the same SSL encoder, terminal epoch, single view (`compare_dev.json`).

| | generalist framed | specialist framed | delta framed [95 % CI] | hybrid with specialist on framed rows, all rows |
|---|---:|---:|---|---|
| fold 0 | 9.572 | 9.622 | +0.051 [-0.080, +0.184] | +0.010 |
| fold 1 | 9.307 | 9.393 | +0.086 [-0.039, +0.212] | +0.013 [-0.017, +0.045] |
| pooled | 9.441 | 9.509 | +0.068 [-0.029, +0.159] | +0.012 [-0.010, +0.035] |

Continue rule (pooled <= -0.15, both folds negative) fails; stop rule (pooled > -0.05) fires. On unframed images the
specialist is +1.97 worse, as expected for a model that never saw them.

Reading: training on the framed style alone makes framed predictions slightly worse, so the unframed images help
the framed ones; the shared model is not compromising between the styles. Together with the masked-pooling result
(framed +0.031), two mechanisms for recipe A's framed-image weakness are now rejected (black-token pooling, style
competition). The cause stays open; the remaining candidates (whole-frame view vs Phase-1's central crops,
device/magnification/eye bank) need changes that cannot be validated before the 11-09 freeze.
