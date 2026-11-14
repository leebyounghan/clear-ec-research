---
id: con-pod-safety
type: constraint
name: Pod and repo safety rules
summary: Never touch jhw0714-*; no du on /workspace (Lustre); no deletions (move to archive); no docker latest tag; never edit a running bash script; waiters are bash scripts with perl alarm.
status: active
origin: user
date: '2026-11-07'
scope: global
source: k8s-server CLAUDE.md; docs/LANES_20261108.md 'Rules that do not change'; memory/never-edit-running-bash-script.md
---

# Pod and repo safety rules

## Rule

Never touch jhw0714-*; no du on /workspace (Lustre); no deletions (move to archive); no docker latest tag; never edit a running bash script; waiters are bash scripts with perl alarm.
