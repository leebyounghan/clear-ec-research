---
id: mccarey2008review
type: paper
title: "Review of corneal endothelial specular microscopy for FDA clinical trials of refractive procedures, surgical devices and new intraocular drugs and solutions"
authors: [Bernard E McCarey, Henry F Edelhauser, Michael J Lynn]
year: 2008
venue: "Cornea"
doi: "10.1097/ICO.0b013e31815892da"
arxiv: ""
pmcid: "PMC3062434"
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC3062434"
citation: "McCarey BE, Edelhauser HF, Lynn MJ. Review of corneal endothelial specular microscopy for FDA clinical trials of refractive procedures, surgical devices and new intraocular drugs and solutions. Cornea. 2008;27(1):1-16."
tier: core
group: t2-device-morphometry
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [pmc]
evidence: review
tldr: "Standard reference on how specular microscopes derive ECD, CV and HEX: the centre-to-centre method needs a cell plus its contiguous neighbours to get an area, so about 150 dotted cells yield about 100 areas; sampling error, cell omissions, off-centre dots and calibration all bias the numbers."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Explains why a label from about 150 dotted cells carries interior-cell areas only, and lists the error sources (dot placement inflates area SD, magnification 1-pixel error about 1.6 % ECD) that make up device behaviour."
code_release: none
weights_release: none
data_scale: "review; cites clinical-trial and reading-centre data"
key_numbers:
  - "Centre method: after dotting 100 cells, area is determined for about 70 %, so about 150 cells must be dotted to get 100 areas (verified-from-source)"
  - "Centre method stated error 0.5-5.0 %; corner method total error about 5 % (verified-from-source)"
  - "Repeat analysis of one pattern (10 runs): CV 41 +- 1, hexagonal 45 +- 2 % (78 cells); CV 48 +- 1, hexagonal 51 +- 1 % (147 cells) (verified-from-source)"
  - "Calibration error of 1 pixel gives about 1.6 % ECD error; 2-3 pixels 3.2-4.8 % (verified-from-source)"
  - "Omitting or double-entering a cell changes density by about 1 % per cell (verified-from-source)"
  - "Off-centred dots increase the SD of cell area (affects CV); no number given (verified-from-source)"
  - "Normal young-adult CV about 0.22-0.31 (mean 0.27, Yee et al.); normal HEX about 60 % (verified-from-source)"
results: []
suggests_try: "Read device CV as 'CV of interior-cell polygon areas from about 150 dotted centres, with dot jitter': in any emulator, drop the outer ring of cells and add dot jitter at the level our detector has (0.2 x spacing) before fitting the map to labels, so the fitted device map is learned under the same noise the test frames will have."
tags: [review, fda, center-method, corner-method, sampling-error, cv, hexagonality, ecd]
search_keywords: ["specular microscopy FDA clinical trials review", "center method cell area neighbours", "number of cells to count endothelium"]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: author, note: "Accuracy of the centre method depends on the algorithm and technician diligence; manufacturers use proprietary software.", locator: "Techniques and Errors section"}
  - {stated_by: reviewer, note: "No CV/HEX inter-observer numbers; no device-specific polygon algorithm; 2008 devices.", locator: "Repeatability section"}
---

# Review of corneal endothelial specular microscopy for FDA clinical trials

## TL;DR
A methods review for FDA trials. It describes comparison, frame, corner and centre-to-centre analysis; recommends cell counts; and quantifies error sources for density (calibration, omissions). For the centre method, an area exists only for cells whose contiguous neighbours are dotted, so about 150 dotted cells give about 100 areas.

## Citation and link
- McCarey BE, Edelhauser HF, Lynn MJ. Cornea 2008;27(1):1-16. PMC3062434.
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3062434

## Problem and motivation
Standardise specular microscopy endpoints (ECD, CV, HEX) for device and drug trials.

## Data
Review; cites Hirst sampling data, Jones et al. Konan ROBO repeatability, Benetz reading-centre data, Cornea Donor Study.

## Method
- Centre-to-centre: operator dots each centre; a nearest-neighbour algorithm (pseudo-inscribed circle and average radius to adjacent centres) computes area; each area needs the cell plus its six contiguous neighbours.
- Corner: operator marks vertices; software computes area and side count.
- Frame / variable frame: counts only, no areas.
- CV = SD(area)/mean(area); HEX = share of six-sided cells.

## Results
See key_numbers (verified-from-source). Sampling error for a single 120-cell sample (Hirst data, density): +-14.4 % homogeneous, +-21.6 % moderate polymegethism, +-27.6 % extreme.

## Limitations
- Author-stated: technician and algorithm dependence.
- Reviewer: no inter-observer CV/HEX numbers.

## Relevance to CD / CV / HEX
- CD: label = 1e6 / mean interior-cell area, matching our label fact.
- CV: interior cells only; dot placement noise inflates area SD, so device CV includes a technician-jitter component (our marker-jitter test: 20 % jitter drops CV r from 0.89 to 0.83).
- HEX: polygon sides from the device construction.

## What it suggests we try
Calibrate any point-based CV/HEX emulator under realistic dot jitter and interior-cell selection, then learn the map to labels; this targets device behaviour rather than cell finding.

## Related notes
- [[takahashi1996konanpatent]] — the actual Konan polygon rule.
- [[doughty2020cvcells]] — how many cells CV needs.
- [[scarpa2016morphometry]] — automatic vs manual on identical images.

## Sources
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3062434
