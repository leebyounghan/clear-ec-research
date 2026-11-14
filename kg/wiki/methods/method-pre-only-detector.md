---
id: method-pre-only-detector
type: method
name: Pre-only cell detector (trained on external Topcon points)
summary: 'Heatmap peak detector trained only on the external Topcon 385 set. On the 25 expert frames: miss 0.150, false 0.165, jitter 0.215 of spacing, F1 0.862. Three to sixteen seeds deployed.'
code: experiments/green_track_20261028/deploy/detectors/ (pod)
literature:
- target: '[[daniel2019unet]]'
  relation: explains
  note: trained on these 385 point images; U-Net recall 0.34 still gave ECD R2 0.96
- target: '[[wang2020dmcount]]'
  relation: suggests
  note: OT + count loss for a density/heatmap head on point annotations
- target: '[[wustefeld2023blobcount]]'
  relation: suggests
  note: CD-derived count in the box as an extra loss on the heatmap
---
# Pre-only cell detector (trained on external Topcon points)

Heatmap peak detector trained only on the external Topcon 385 set. On the 25 expert frames: miss 0.150, false 0.165, jitter 0.215 of spacing, F1 0.862. Three to sixteen seeds deployed.

## Where it is used

Experiments link here with `uses_method`; see backlinks or `kg_query.py node method-pre-only-detector`.
