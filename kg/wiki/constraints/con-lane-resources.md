---
id: con-lane-resources
type: constraint
name: One GPU per lane, own folder, at most 4 CPU workers
summary: Only your GPU and lane folder, at most 4 CPU workers; kill only verified pids; no idle GPU (keepalive last resort).
status: active
origin: claude
date: '2026-11-07'
scope: global
source: docs/LANES_20261108.md 'Rules that do not change'
---

# One GPU per lane, own folder, at most 4 CPU workers

## Rule

Only your GPU and lane folder, at most 4 CPU workers; kill only verified pids; no idle GPU (keepalive last resort).
