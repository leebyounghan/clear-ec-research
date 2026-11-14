---
id: con-no-runtime-caveats
type: constraint
name: No T4 or per-image runtime caveats in rules or reports
summary: Do not turn the official runtime limit into a lane rule or report caveat; runtime is checked once on the package track.
status: active
origin: user
date: '2026-11-08'
scope: global
source: memory/clear-ec-no-runtime-caveats.md
---

# No T4 or per-image runtime caveats in rules or reports

## Rule

Do not turn the official runtime limit into a lane rule or report caveat; runtime is checked once on the package track.

## Package-track check, 2026-11-12 / 13

The one runtime check this rule leaves to the package track was made: [[con-platform-runtime-300s-t4]] holds the measured times (tier p 287-314 s on two Colab T4 VMs, t3 173-174 s; the platform T4 itself unmeasured). The PI's own instruction of 11-12 was to assume the T4 (D-PKG-006).
