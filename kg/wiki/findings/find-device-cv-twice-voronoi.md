---
id: find-device-cv-twice-voronoi
type: finding
name: Device CV is about twice the Voronoi area CV of the same markers
summary: The device's CV is about 2x the area CV of a Voronoi on the same expert markers.
status: robust
metrics:
- CV
date: '2026-11-08'
source: 'docs/BRIEF.md ''Measured facts: Labels'''
literature:
- target: '[[bucht2011simulation]]'
  relation: corroborates
  note: simulated images exposed a scaling error in the Topcon software's CV estimate
- target: '[[sierra2023sdm]]'
  relation: corroborates
  note: Topcon software CV +29.9 points above manual, rising with CV (device behaviour; this part is not superseded)
- target: '[[takahashi1996konanpatent]]'
  relation: explains
  note: 'weighted bisectors amplify area spread vs Voronoi (k=2 fitted: scale 1.09)'
- target: '[[gasser2015konantopcon]]'
  relation: corroborates
  note: 'Konan CV 37 vs Topcon 29 on the same eyes: the device construction sets the CV scale'
- target: '[[patel2010flexcenter]]'
  relation: explains
  note: analysis mode alone shifts CV ~10 % relative on the same cells
---
# Device CV is about twice the Voronoi area CV of the same markers

The device's CV is about 2x the area CV of a Voronoi on the same expert markers.

## Evidence

Experiments that support or contradict this finding link here (`supports_finding`, `contradicts_finding`); see backlinks.

## Explained 2026-11-09

The ~2x is the device's outline rule: Konan weighted bisectors with k=2 bring the device/estimate scale from 2.08 (Voronoi) to 1.09 on the 25 marker frames ([[exp-cv-konan-bisectors-markers]], [[find-konan-rule-explains-device-cv]]).
