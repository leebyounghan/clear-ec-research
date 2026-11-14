---
id: method-mask-emulator
type: method
name: Device emulator from masks
summary: Compute CD, CV and HEX the way the device would from Cellpose-SAM masks (ok cells / interior cells / Voronoi rules; whole frame, legible window or matched box), calibrate, add to the best pipeline.
code: experiments/rootcause_20261109/source/q2_emulator_20261109.py (pod)
builds_on:
- '[[method-cellpose-sam-masks]]'
---

# Device emulator from masks

Compute CD, CV and HEX the way the device would from Cellpose-SAM masks (ok cells / interior cells / Voronoi rules; whole frame, legible window or matched box), calibrate, add to the best pipeline.

## Where it is used

Experiments link here with `uses_method`; see backlinks or `kg_query.py node method-mask-emulator`.
