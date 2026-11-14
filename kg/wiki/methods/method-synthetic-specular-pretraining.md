---
id: method-synthetic-specular-pretraining
type: method
name: Pretraining on synthetic specular frames with exact labels
summary: Rendered endothelium mosaics (synth_v3, about 49k-50k frames, exact CD / area CV / HEX) used to pretrain an encoder after the fold's SSL and before the unchanged regression recipe; also read zero-shot.
uses_dataset:
- '[[ds-synthetic-specular-v3]]'
---

# Pretraining on synthetic specular frames with exact labels

Rendered endothelium mosaics (synth_v3, about 49k-50k frames, exact CD / area CV / HEX) used to pretrain an encoder after the fold's SSL and before the unchanged regression recipe; also read zero-shot.

## Where it is used

Experiments link here with `uses_method`; see backlinks or `kg_query.py node method-synthetic-specular-pretraining`.

## Status 2026-11-13

All three routes (DINOv3-B, ViT-L, recipe A) are closed for all three metrics: [[find-synthetic-pretraining-gains-vanish-in-stacks]]. Recipe: the same 50k synth_v3 frames, one pass (3,062 updates for recipe A) from the fold's SSL, then the unchanged regression recipe.
