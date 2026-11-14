---
id: method-packaged-deploy-entry
type: method
name: Packaged deploy entry (fp16 copies upcast at load, skipped base weights, per-stage unloading)
summary: 'The Grand Challenge entry of the frozen candidate: fp16 copies of the regression members upcast to fp32 at load, pretrained base weights built from config instead of read, one cpsam_v2 copy, models unloaded between stages (GPU peak 17.35 -> 4.33 GB), one image per fresh process, per-image fallback to submission-1 constants.'
code: lane_cellroute_20261109/inference/pack_hooks.py; inference/docker/gc_entry.py; inference/recipe_infer.py
---

# Packaged deploy entry (fp16 copies upcast at load, skipped base weights, per-stage unloading)

The Grand Challenge entry of the frozen candidate: fp16 copies of the regression members upcast to fp32 at load, pretrained base weights built from config instead of read, one cpsam_v2 copy, models unloaded between stages (GPU peak 17.35 -> 4.33 GB), one image per fresh process, per-image fallback to submission-1 constants.

## Where it is used

Experiments link here with `uses_method`; see backlinks. Sources: ledger O-DEPLOY-012, A-DEPLOY-006, O-DEPLOY-015; `lane_cellroute_20261109/inference/pack_hooks.py`, `inference/docker/gc_entry.py`.
