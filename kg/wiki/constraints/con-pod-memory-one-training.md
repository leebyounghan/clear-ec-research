---
id: con-pod-memory-one-training
type: constraint
name: One training at a time per memory budget; check cgroup memory first
summary: The pod was OOM-killed 2026-11-04; never index an NpzFile member inside a loop; check memory.current before launching a second training.
status: active
origin: claude
date: '2026-11-04'
scope: global
source: memory/clear-ec-pod-oom-two-trainings.md
---

# One training at a time per memory budget; check cgroup memory first

## Rule

The pod was OOM-killed 2026-11-04; never index an NpzFile member inside a loop; check memory.current before launching a second training.
