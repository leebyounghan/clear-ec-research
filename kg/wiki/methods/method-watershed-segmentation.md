---
id: method-watershed-segmentation
type: method
name: Watershed cell boundaries
summary: Watershed on detector or image evidence to get cell boundaries (abs_tess pilot grid); closed for CV and HEX.
literature:
- target: '[[viguerasguillen2019fcn]]'
  relation: source-of-method
  note: U-net edges + Fourier-scale watershed on Topcon SP-1P
- target: '[[kucharski2021cnnwatershed]]'
  relation: source-of-method
  note: CNN markers + marker-controlled watershed
- target: '[[viguerasguillen2018merging]]'
  relation: explains
  note: oversegment-then-merge with an SVM; 2.0 % undersegmented
---
# Watershed cell boundaries

Watershed on detector or image evidence to get cell boundaries (abs_tess pilot grid); closed for CV and HEX.

## Where it is used

Experiments link here with `uses_method`; see backlinks or `kg_query.py node method-watershed-segmentation`.
