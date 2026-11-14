---
id: method-recipe-a
type: method
name: Recipe A (CPDINO ViT-B whole frame)
aliases:
- A
- A2
- CPDINO
summary: 'Cellpose-pretrained DINO ViT-B (kernel 16, stride 8) on the whole frame: SSL5 (LeJEPA, 5 epochs), std pooling, schedule10, four flips, 20 epochs, 5 folds x 2 seeds. The winner every family is compared with.'
code: experiments/meta_ceiling_20261103/source/phase2_foundation_train_20261002.py (pod)
literature:
- target: '[[liang2022transcrowd]]'
  relation: explains
  note: 'count-only ViT regression: GAP over tokens beats a regression token'
---
# Recipe A (CPDINO ViT-B whole frame)

Cellpose-pretrained DINO ViT-B (kernel 16, stride 8) on the whole frame: SSL5 (LeJEPA, 5 epochs), std pooling, schedule10, four flips, 20 epochs, 5 folds x 2 seeds. The winner every family is compared with.

## Details

Member read on folds 0 and 4: 9.046 (seed 46, epochs 16-20); 9.041 (seeds 46-48, epochs 16-20); 9.0166 (seed 46, epochs 8-12). Every recipe A change tested end to end came back to the same solution (residual r 0.98-0.99).

## Where it is used

Experiments link here with `uses_method`; see backlinks or `kg_query.py node method-recipe-a`.
