---
id: method-cell-graph-network
type: method
name: Cell-graph network (GNN on detector peaks)
summary: Voronoi graph of pre-only detector peaks in the three windows with most confident peaks, three detectors (pre46/47/48); graph network (d64, 5 layers, 25 epochs) on CV and HEX. In the best pipeline it adds CV -0.046.
code: experiments/meta_ceiling_20261103/source/cell_graph_sweep3_20261105.py (pod)
builds_on:
- '[[method-pre-only-detector]]'
literature:
- target: '[[shumilin2023voronoi]]'
  relation: suggests
  note: a differentiable Voronoi head (points -> areas); judged not worth building
- target: '[[piorkowski2017segmentation]]'
  relation: explains
  note: side-length / neighbour-distance CV is robust to segmentation method
---
# Cell-graph network (GNN on detector peaks)

Voronoi graph of pre-only detector peaks in the three windows with most confident peaks, three detectors (pre46/47/48); graph network (d64, 5 layers, 25 epochs) on CV and HEX. In the best pipeline it adds CV -0.046.

## Where it is used

Experiments link here with `uses_method`; see backlinks or `kg_query.py node method-cell-graph-network`.
