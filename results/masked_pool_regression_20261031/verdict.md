# masked_pool_regression_20261031 — verdict (fold 4, 2026-10-31 ~19:40 UTC)

Pre-registered continue rule (protocol.json): framed delta <= -0.15 and unframed delta <= +0.05. FAILED.

| fold 4, seed 46, single view | standard std | masked std | delta [95 % CI] |
|---|---:|---:|---:|
| framed images (n 860) | 9.421 | 9.452 | +0.031 [-0.069, +0.128] |
| unframed images (n 930) | 8.703 | 8.635 | -0.068 [-0.144, +0.006] |
| all | 9.048 | 9.027 | -0.020 [-0.086, +0.045] |
| hybrid (0.5 member x LOFO scalars + 0.5 Phase-1 five), all | | | +0.001 [-0.031, +0.034] |

Reading: excluding black-frame tokens from the std pooling does not recover recipe A on framed images, so the
mixing of background tokens is not why recipe A loses its advantage there (the normalisation window was already
ruled out). The cause stays open (candidates: acquisition device / magnification or eye-bank differences that
co-occur with the frame, not the frame pixels themselves). Fold 0 (GPUs 2-3) is allowed to finish because the
GPUs would otherwise idle; by the protocol it is reported descriptively and cannot change this verdict.
Masked pooling does not enter submission 2.

## Fold 0 (descriptive only, finished 2026-10-31 21:08 UTC)
framed -0.016 [-0.113, +0.078]; unframed +0.029; all +0.007; hybrid +0.015. Consistent with the fold-4 rejection.
