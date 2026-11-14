---
id: method-segmenter-self-training
type: method
name: Segmenter self-training with label-filtered pseudo-labels
summary: Teacher-student adaptation of the segmenter on frames whose mask-derived CD and HEX agree with the image labels (within 5 % or 10 %).
literature:
- target: '[[reith2025selfadapt]]'
  relation: source-of-method
- target: '[[tyagi2025boxcell]]'
  relation: suggests
  note: pseudo-masks from prompts; detector errors propagate
---

# Segmenter self-training with label-filtered pseudo-labels

Teacher-student adaptation of the segmenter on frames whose mask-derived CD and HEX agree with the image labels (within 5 % or 10 %).

## Where it is used

Experiments link here with `uses_method`; see backlinks or `kg_query.py node method-segmenter-self-training`.
