---
id: con-gpu-not-idle
type: constraint
name: Do not let GPUs idle (reclaim below 10 % use)
summary: The provider reclaims GPUs below 10 % utilisation; queue the next real job before the current one ends.
status: active
origin: user
date: '2026-10-01'
scope: global
source: k8s-server CLAUDE.md; memory/gpu-reclaim-keep-busy.md
---

# Do not let GPUs idle (reclaim below 10 % use)

## Rule

The provider reclaims GPUs below 10 % utilisation; queue the next real job before the current one ends.

## Suspended for the CLEAR-EC pod on the evening of 2026-11-13

PI order of 21:40 EST 11-13 (ledger D-GPU-002): everything on GPUs 0-3 of pod clear-ec-research-4gpu-1104 was taken down and no lane starts a keepalive or any GPU job from then on; for this pod that superseded the keep-busy rule ([[dec-021-gpu-shutdown-closeout-20261113]], [[con-no-gpu-jobs-after-20261113-shutdown]]). That order was itself lifted the same evening for a CD training-setting screen on all four GPUs (ledger line 211). The rule itself stays active for the cluster. Between 11-12 and 11-13 idle cards had been held by a yielding keepalive (about 35 % duty, 1.2 GB) and by one real run started for that purpose ([[exp-bb-synth-recipe-a-seed47-fold0]]).
