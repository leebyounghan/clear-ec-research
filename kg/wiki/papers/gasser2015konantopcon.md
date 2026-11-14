---
id: gasser2015konantopcon
type: paper
title: "Comparison of corneal endothelial cell measurements by two non-contact specular microscopes"
authors: [Lorenz Gasser, Thomas Reinhard, Daniel Böhringer]
year: 2015
venue: "BMC Ophthalmology"
doi: "10.1186/s12886-015-0068-1"
arxiv: ""
pmcid: "PMC4517400"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC4517400/"
citation: "Gasser L, Reinhard T, Böhringer D. Comparison of corneal endothelial cell measurements by two non-contact specular microscopes. BMC Ophthalmol. 2015;15:87. doi:10.1186/s12886-015-0068-1"
tier: core
group: t2-device-morphometry
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: peer-reviewed
tldr: "With cells marked manually by one examiner on both devices, the Konan Noncon Robo SP8000 read ECD 187 cells/mm2 (9.6 %) higher and CV 8 points higher (37 vs 29) than the Topcon SP3000P on the same 34 eyes; CV and HEX correlated only weakly between devices."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Direct evidence that the CV scale is set by the device's own polygon construction, not by the cells; our labels (likely Konan) and the external Topcon point set live on different CV scales."
code_release: none
weights_release: none
data_scale: "34 eyes of 18 patients; Konan 110 +- 45 and Topcon 100 +- 30 cells per measurement"
key_numbers:
  - "ECD Konan 2252 +- 704 vs Topcon 2065 +- 657 cells/mm2; mean difference 187, mean deviation +9.6 %; Konan higher in 32/34 eyes (verified-from-source)"
  - "ECD Spearman rho 0.94 between devices (verified-from-source)"
  - "CV Konan 37 +- 6 vs Topcon 29 +- 5 (p < 0.001); 11/34 eyes differ by > 10 points; correlation 'weak' (no coefficient) (verified-from-source)"
  - "HEX Konan mean 56 (45-70) vs Topcon mean 55 (0-100), p = 0.82; 8/34 eyes differ by > 10 points (verified-from-source)"
  - "Cells were marked manually on both devices by one examiner; built-in software computed ECD, CV, HEX (verified-from-source)"
results:
  - {task: "device agreement", dataset: "34 clinic eyes", metric: "CV mean", value: "37 (Konan)", comparator: "Topcon SP3000P", comparator_value: "29", locator: "Results table", provenance: "verified-from-source"}
  - {task: "device agreement", dataset: "34 clinic eyes", metric: "ECD mean", value: "2252 (Konan)", comparator: "Topcon SP3000P", comparator_value: "2065", locator: "Results table", provenance: "verified-from-source"}
suggests_try: "Never feed CV or HEX computed on the external Topcon set as targets without a device map: use Topcon points only to train the point detector, and learn the Konan-scale CV from our own labels through a device emulator (weighted-bisector polygons) or a fitted monotone map; check that the map's slope is not 1."
tags: [konan, topcon, device-agreement, cv, hexagonality, ecd]
search_keywords: ["Konan Topcon specular microscope agreement", "non-contact specular microscopes interchangeability", "coefficient of variation device bias"]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Agreement may depend on cell density and regularity; small sample.", locator: "Discussion, limitations"}
  - {stated_by: reviewer, note: "Image area and the devices' polygon algorithms are not reported; no CV/HEX correlation coefficient or Bland-Altman limits given; clinic eyes, not donor corneas.", locator: "Methods/Results"}
---

# Comparison of corneal endothelial cell measurements by two non-contact specular microscopes

## TL;DR
One examiner marked cells manually on both a Konan SP8000 and a Topcon SP3000P for 34 eyes. Konan read higher ECD (+9.6 %) and much higher CV (37 vs 29), and CV and HEX agreed poorly between devices, so the authors advise against using them interchangeably.

## Citation and link
- Gasser L, Reinhard T, Böhringer D. BMC Ophthalmol 2015;15:87. doi:10.1186/s12886-015-0068-1
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4517400/

## Problem and motivation
Clinics switch or mix specular microscopes; are values interchangeable?

## Data
34 eyes of 18 consecutive cornea-clinic patients (ages 23-82). Konan Noncon Robo SP8000 and Topcon SP3000P with auto-alignment. Cells analysed: Konan 110 +- 45, Topcon 100 +- 30 per measurement.

## Method
Manual cell marking on both devices (Konan: all clearly visible cells; Topcon: until the software stopped accepting marks); each device's built-in software computed ECD, CV, HEX. Spearman correlation and difference counts.

## Results
See key_numbers (verified-from-source). The CV gap of 8 points is about 25 % relative, with the cells chosen and marked by the same person.

## Limitations
- Author-stated: small sample; dependence on density/regularity possible.
- Reviewer: polygon algorithms undocumented; no LoA for CV/HEX.

## Relevance to CD / CV / HEX
- CV: the device's construction from marked cells, not the cells, sets most of the CV scale (Konan reads higher; consistent with the weighted-bisector rule in [[takahashi1996konanpatent]]). Our "device CV about 2x Voronoi CV" fact fits this.
- CD: Konan about 9.6 % higher than Topcon; a CD model trained or calibrated on Topcon points needs a scale correction.
- HEX: similar means but weak per-eye agreement; HEX label noise from device behaviour is large.

## What it suggests we try
Keep the external Topcon set as detector training data only; derive Konan-scale CV/HEX from points with an emulator and calibrate on our labels; expect a slope different from 1 between Topcon-style and label CV.

## Related notes
- [[desanctis2006twodevices]] — Topcon vs Konan center method: CV/HEX r < 0.20 between devices.
- [[takahashi1996konanpatent]] — Konan centre-method polygon rule.
- [[bucht2011simulation]] — Topcon program CV scaling error on simulated images.

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4517400/
