---
id: con-cpu-pool
type: constraint
name: Parallelise CPU passes with a pool sized to free quota
summary: CPU-only passes over the 8954 frames are parallelised (pool sized to free cgroup quota, not all 16 cores), vectorised and cached.
status: active
origin: user
date: '2026-11-03'
scope: global
source: memory/pod-cpu-passes-use-pool-of-4.md; memory/clear-ec-mlops-review-20261106.md
---

# Parallelise CPU passes with a pool sized to free quota

## Rule

CPU-only passes over the 8954 frames are parallelised (pool sized to free cgroup quota, not all 16 cores), vectorised and cached.
