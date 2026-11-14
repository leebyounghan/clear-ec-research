---
id: con-no-gpu-jobs-after-20261113-shutdown
type: constraint
name: No keepalive and no GPU job on the pod after the shutdown of 2026-11-13 21:41 EST (lifted the same evening)
summary: 'PI order of 2026-11-13 21:40 EST: everything on GPUs 0-3 was taken down and no lane was to start a keepalive or any GPU job; for this pod it superseded the standing keep-busy rule. Lifted the same evening: the ledger records a CD training-setting screen launched on all four GPUs at 22:34 EST with the words ''D-GPU-002 lifted'' (PI via coordinator).'
status: lifted
origin: user
date: '2026-11-13'
scope: global
source: shared notes ledger_20261111.jsonl D-GPU-002 (line 206), D-CD-TRAIN-SCREEN (line 211), D-CD-PRETRAIN-SCREEN (line 213); docs/FINAL_SUMMARY_20261113.md section 0
---

# No keepalive and no GPU job on the pod after the shutdown of 2026-11-13 21:41 EST (lifted the same evening)

PI order of 2026-11-13 21:40 EST: everything on GPUs 0-3 was taken down and no lane was to start a keepalive or any GPU job; for this pod it superseded the standing keep-busy rule. Lifted the same evening: the ledger records a CD training-setting screen launched on all four GPUs at 22:34 EST with the words 'D-GPU-002 lifted' (PI via coordinator).

## Rule

From 21:41 EST 11-13: no lane starts a keepalive or any GPU job on pod clear-ec-research-4gpu-1104 (D-GPU-002, by the PI's own instruction).

## Why

The PI ordered the shutdown and the close-out ([[dec-021-gpu-shutdown-closeout-20261113]]). The keep-busy rule ([[con-gpu-not-idle]]) exists to prevent reclaim of idle cards; the PI's order replaced it for this pod.

## Lifted

Ledger line 211 (D-CD-TRAIN-SCREEN, stamped 22:36 EST 11-13): 'Launched 22:34 EST (PI via coordinator 22:50 EST, D-GPU-002 lifted)'. The coordinator time 22:50 is later than the line's own stamp and than the launch time; it is kept as written. No separate decision line records the lifting, and its scope beyond that screen is not stated. What runs under it: [[exp-cd-training-setting-screen-dinov3]] and, from 22:43 EST, [[exp-cd-ssl-pretrain-screen-dinov3]]. The consolidated summary (22:15 EST) still says that all four GPUs are down and no lane process is running; the ledger line is later.
