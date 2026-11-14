---
id: method-best-pipeline
type: method
name: Best pipeline (20261106)
summary: Hybrid B with hex6 on HEX, log-affine stretch, the cell-graph network on CV and HEX and a ridge on a graph encoder pretrained on simulated tessellations. OOF 8.739 on five folds; the reference every lane measures against.
code: experiments/meta_ceiling_20261103/source/best_pipeline_20261106.py (pod)
builds_on:
- '[[method-hybrid-b]]'
- '[[method-hex6]]'
- '[[method-log-affine-stretch]]'
- '[[method-cell-graph-network]]'
- '[[method-sim-graph-encoder-ridge]]'
---

# Best pipeline (20261106)

Hybrid B with hex6 on HEX, log-affine stretch, the cell-graph network on CV and HEX and a ridge on a graph encoder pretrained on simulated tessellations. OOF 8.739 on five folds; the reference every lane measures against.

## Details

BRIEF: 'Best pipeline, not yet in an image'. Screen rows (folds 0 and 4, 3581 rows) read 8.7578 on the mean (LANES).

## Where it is used

Experiments link here with `uses_method`; see backlinks or `kg_query.py node method-best-pipeline`.
