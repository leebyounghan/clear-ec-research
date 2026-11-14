---
id: method-deploy-tiers
type: method
name: Deploy tiers by build argument (full, p, t1, t2, t3)
summary: 'The frozen candidate packaged at five sizes from one Dockerfile (ARG TIER): each tier is a set of per-metric parameter files refit with the nested procedure and no outer fold (L5), a manifest of the weights it loads, and a cell-route refit on that tier''s HEX OOF. p = CD t1 + CV t1 + HEX t2 + cell-route on t2.'
code: lane_cellroute_20261109/inference/docker/Dockerfile.freeze (ARG TIER=full|p|t1|t2|t3); inference/params_build.py
---

# Deploy tiers by build argument (full, p, t1, t2, t3)

The frozen candidate packaged at five sizes from one Dockerfile (ARG TIER): each tier is a set of per-metric parameter files refit with the nested procedure and no outer fold (L5), a manifest of the weights it loads, and a cell-route refit on that tier's HEX OOF. p = CD t1 + CV t1 + HEX t2 + cell-route on t2.

## Where it is used

Experiments link here with `uses_method`; see backlinks. Sources: ledger D-PKG-002, D-PKG-003, A-DEPLOY-007; `lane_cellroute_20261109/inference/docker/{Dockerfile.freeze, stage_package.py, tier_table.json}`.
