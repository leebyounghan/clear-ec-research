---
id: find-lit-topcon-konan-cv-scale
type: finding
name: 'Literature: Topcon and Konan read different CV on the same cells (37 vs 29)'
summary: On 34 eyes with manually marked cells on both devices (gasser2015konantopcon), Konan CV 37 +- 6 vs Topcon 29 +- 5 (p < 0.001), HEX 56 vs 55 (p = 0.82), ECD Konan +9.6 %. The CV scale is set by the device's polygon construction, so Topcon-device CV/HEX values are not on the CLEAR-EC (Konan-like) label scale.
status: robust
metrics:
- CV
- HEX
- CD
date: '2026-11-09'
source: paper note gasser2015konantopcon (key_numbers, verified from source)
contradicts:
- target: '[[idea-external-author-access-data]]'
  note: 'assumption contradicted: that labelled Topcon SP-1P data (Rotterdam) can serve as extra CV/HEX targets as they are; their CV is on another device scale (and ECD ~9.6 % lower). Points or masks transfer; device CV/HEX values need a device map.'
evidence_for:
- target: '[[find-device-cv-twice-voronoi]]'
  note: the device's polygon construction sets the CV scale
- target: '[[find-konan-rule-explains-device-cv]]'
  note: Konan's weighted-bisector rule reproduces our label scale (1.09)
qualifies:
- target: '[[find-external-topcon-scale-shift]]'
  note: a CD scale difference between Topcon frames and our labels is expected (Konan ECD +9.6 % on the same eyes)
literature:
- target: '[[gasser2015konantopcon]]'
  relation: source
  note: CV 37 vs 29; HEX 56 vs 55; ECD 2252 vs 2065
---
# Literature: Topcon and Konan read different CV on the same cells (37 vs 29)

On 34 eyes with manually marked cells on both devices (gasser2015konantopcon), Konan CV 37 +- 6 vs Topcon 29 +- 5 (p < 0.001), HEX 56 vs 55 (p = 0.82), ECD Konan +9.6 %. The CV scale is set by the device's polygon construction, so Topcon-device CV/HEX values are not on the CLEAR-EC (Konan-like) label scale.

## Evidence

- gasser2015konantopcon: ECD Konan 2252 +- 704 vs Topcon 2065 +- 657 (mean deviation +9.6 %, Konan higher in 32/34 eyes); CV Konan 37 +- 6 vs Topcon 29 +- 5 (p < 0.001), 11/34 eyes differ by > 10 points; HEX 56 vs 55 (p = 0.82); one examiner marked cells on both devices.

## Limits

34 eyes, one examiner. CLEAR-EC's device is not published (KONAN is named as an example, clearec2026challenge); the Konan-rule fit ([[exp-cv-konan-bisectors-markers]], scale 1.09) is our evidence that labels behave Konan-like.

## Bears on

[[ds-topcon-external-385]] is used for points only (detector training), which stays valid. Any external Topcon CV/HEX target needs the device map first ([[idea-external-author-access-data]], [[dec-009-author-access-request]]).
