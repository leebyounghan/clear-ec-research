---
id: method-cellpose-sam-masks
type: method
name: Zero-shot Cellpose-SAM masks (seg_loose)
summary: 'Cellpose-SAM v2 run zero-shot with loose thresholds (flow 0.8, cellprob -2) and CLAHE on all 8954 frames: 85-90 % of frames usable (default 53 %). Cell-shape features, Delaunay counts and mask area feed HEX and CD.'
code: experiments/lane_hex_20261108/source/seg_hex.py, seg_run.py; masks in seg_loose/ (pod)
literature:
- target: '[[pachitariu2025cellposesam]]'
  relation: source-of-method
  note: no endothelium/specular data in its training set (paper note)
- target: '[[stringer2025cellpose3]]'
  relation: suggests
  note: restoration against a segmentation loss; does not model vignetting
- target: '[[stringer2021cellpose]]'
  relation: explains
  note: flow representation; diameter is the main knob for ~25-30 px cells
---
# Zero-shot Cellpose-SAM masks (seg_loose)

Cellpose-SAM v2 run zero-shot with loose thresholds (flow 0.8, cellprob -2) and CLAHE on all 8954 frames: 85-90 % of frames usable (default 53 %). Cell-shape features, Delaunay counts and mask area feed HEX and CD.

## Where it is used

Experiments link here with `uses_method`; see backlinks or `kg_query.py node method-cellpose-sam-masks`.
