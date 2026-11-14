---
id: idea-cd-technician-window-correction
type: idea
name: 'CD: technician-window density, a density slope, and a CV-conditioned CD correction'
summary: Read density inside a predicted technician window, let the CD correction slope grow with density, and correct CD by predicted CV / cell size. The window and slope parts are already covered (window mask density = plain mean; the log-affine stretch fits a slope); only the CV-conditioned correction is untested.
status: open
origin: literature
aims_at:
- CD
owner_lane: cd
cost: ~1 CPU-h (nested ridge on stored CD and CV predictions)
cost_hours: 1
cost_unit: CPU-h
potential_metric: CD
potential_gain: 0.003
potential_basis: 'estimate: residual-type corrections on the CD stack read 0.000 to -0.003 lately (Voronoi CD +0.000, recipe A s48 -0.003); the device bias by CV and cell size has no effect size in huang2017autovsmanual''s abstract'
source: Literature review 2026-11-09 (paper notes' suggests_try), coordinator list of eight suggestions, suggestion 5
novelty_check: 'KG check 2026-11-09 (kg_query check + grep over wiki/): check window cd; check density; check stretch -> exp-cd-mask-density-variants (densest window -0.018 = plain mean, closed), exp-cd-detector-density-global (weight 0), exp-cd-mask-voronoi-cd (+0.000), method-log-affine-stretch; the CD lane''s own KG check on 11-09 (line 67) also calls the window part covered. Literature: papers from the KG literature review (benetz2024donorecd, huang2017autovsmanual, daniel2019unet, lempitsky2010counting); no new web search in this pass.'
motivated_by:
- '[[find-technician-region]]'
- '[[find-mask-cd-direct-density]]'
- '[[find-cd-label-definition]]'
builds_on:
- '[[method-mask-density-cd]]'
- '[[method-log-affine-stretch]]'
- '[[method-nested-blend]]'
tested_by:
- target: '[[exp-cd-mask-density-variants]]'
  closeness: same
  note: 'window part: densest-window mask density -0.018 = plain mean (closed); the CD lane did not rerun it (line 67)'
- target: '[[exp-cd-detector-density-global]]'
  closeness: close
  note: 'detector density as a component: weight 0'
- target: '[[exp-cd-mask-voronoi-cd]]'
  closeness: close
  note: '1/mean Voronoi area of mask centroids: same information as mask area'
- target: '[[exp-ref-best-pipeline-5fold]]'
  closeness: related
  note: 'slope part: the per-metric log-affine stretch (nested) already fits a density slope'
literature:
- target: '[[benetz2024donorecd]]'
  relation: source-of-method
  note: region matched to the eye-bank area halves the bias; bias grows ~120-134 cells/mm2 per 500 cells/mm2
- target: '[[huang2017autovsmanual]]'
  relation: source-of-method
  note: device ECD bias depends on age/CV and cell size
- target: '[[daniel2019unet]]'
  relation: suggests
  note: 1/median Voronoi area of confident detections, robust at recall 0.34
- target: '[[lempitsky2010counting]]'
  relation: suggests
  note: density integral over a region
---
# CD: technician-window density, a density slope, and a CV-conditioned CD correction

## Hypothesis

CD label error grows with density and with CV (polymegethous corneas); a CD correction on (predicted CD, predicted CV) fitted by nested OOF MAPE absorbs it.

## Closest tries in the KG

- Window: [[exp-cd-mask-density-variants]] (same, closed).
- Density as a component: [[exp-cd-detector-density-global]], [[exp-cd-mask-voronoi-cd]] (close, nothing added).
- Slope: the log-affine stretch in [[exp-ref-best-pipeline-5fold]] (related).
- CV-conditioned correction: no try in the KG ([[exp-cv-residual-from-hex-cd]] is the reverse direction, CV from CD/HEX: +0.135).

## Literature

benetz2024donorecd, huang2017autovsmanual, daniel2019unet, lempitsky2010counting.

## Minimal test

Nested five-fold ridge of log CD residual on (log predicted CD, predicted CV, their product) on top of the current CD stack (6.1775); keep if CD OOF goes down. No per-source terms ([[con-no-source-id-shortcuts]]).

## Risks

CV predictions are weak (r 0.686) and share errors with CD; the correction may only re-fit the stretch.
