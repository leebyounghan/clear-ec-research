---
id: clearec2026challenge
type: paper
title: "CLEAR-EC: Corneal Learning for Endothelial Assessment and Review using AI for Endothelial Count (challenge site, dataset record and baseline)"
authors: [Lama Al-Aswad, Peixian Liang, Zhi Huang]
year: 2026
venue: "website (grand-challenge.org, Zenodo, GitHub, OPTIC Lab)"
doi: "10.5281/zenodo.21270595"
arxiv: ""
pmcid: ""
url: "https://clear-ec.grand-challenge.org/"
citation: "Al-Aswad L, Liang P, Huang Z, Eye-Bank for Sight Restoration, AltruVision (Lions Eye Bank), The Eye Bank of Kentucky, Georgia Eye Bank. CLEAR-EC: Corneal Learning for Endothelial Assessment and Review using AI for Endothelial Count [dataset]. Zenodo; 2026. doi:10.5281/zenodo.21270595"
tier: core
group: t3-challenge
status: reviewed
review_depth: deep
reviewed_on: 2026-11-09
sources_read: [website, repo]
evidence: website
tldr: "Public pages state the task, metric and data origin (9,000 training + 1,000 test specular images from US eye banks in NY, GA, KY, DE) but not which device or analysis mode produced the CD/CV/HEX labels; the baseline is Cellpose v1 'cyto' plus per-cell morphology in a random 40 % x 40 % crop."
relevance_clearec: high
relevance_metrics: [CD, CV, HEX]
relevance_note: "Primary source for what the organisers say about labels: nothing about the device algorithm is published; KONAN is named only as an example of the eye-bank workflow."
code_release: open
weights_release: none
data_scale: "9,000 training + 1,000 test images (BMP, about 1.5-2 MB); Phase I 100 test images, Phase II 1,000"
key_numbers:
  - "Score: per-metric mean absolute percent error |pred - gt| / gt x 100, equal-weight mean of CD, CV, HEX (verified-from-source)"
  - "9,000 training and 1,000 test images from eye banks in New York, Georgia, Kentucky and Delaware (verified-from-source)"
  - "Phase I: 100 test images, up to 3 runs; Phase II: 1,000 test images, up to 2 runs; Phase II Oct 1 - Nov 14 (verified-from-source)"
  - "Baseline pixel size fixed at 1000/1296 um per pixel; CD = N x 1e6 / total area; CV = std/mean area; baseline HEX = mean IoU of each cell with a fitted hexagon (not the six-sided share) (verified-from-source)"
  - "Device and label-generation procedure: not reported on any public page (verified-from-source)"
results: []
suggests_try: "Since the organisers name KONAN as the eye-bank device and give no label algorithm, test the Konan centre-method emulator (weighted bisectors, interior cells) against the labels before assuming Voronoi or mask statistics; the baseline's hexagon-IoU HEX is not the label definition and should not be copied."
tags: [challenge, grand-challenge, eye-bank, dataset, baseline, cellpose]
search_keywords: ["CLEAR-EC challenge", "corneal endothelial cell count AI challenge eye bank", "Al-Aswad specular microscopy AI"]
uses_dataset: []
uses_method: []
compares_with: []
extends: []
has_limitation:
  - {stated_by: reviewer, note: "No published description of how labels were produced (device model, center vs flex-center vs auto, technician protocol); no challenge paper; Phase I/II algorithm descriptions are behind sign-in and were not read.", locator: "challenge home, Zenodo, GitHub README"}
---

# CLEAR-EC challenge (site, dataset record, baseline)

## TL;DR
Penn OPTIC Lab (Al-Aswad) with Huang lab (Penn Pathology) and four US eye banks run CLEAR-EC to standardise donor endothelial assessment. The public material defines the task and the metric, the data origin and the timeline, but not how the labels were computed. The organisers' baseline segments with Cellpose v1 and computes metrics in a random crop.

## Citation and link
- Zenodo dataset doi:10.5281/zenodo.21270595 (restricted; CLEAR-EC Restricted Research Data License v1.0; v1 published 2026-07-09, modified 2026-07-16). Creators: Al-Aswad L, Liang P, Huang Z (University of Pennsylvania); Eye-Bank for Sight Restoration; AltruVision (Lions Eye Bank); The Eye Bank of Kentucky; Georgia Eye Bank.
- https://clear-ec.grand-challenge.org/ ; https://opticlab.med.upenn.edu/clear-ec-challenge ; https://github.com/CLEAR-EC/CLEAR-EC ; https://zenodo.org/records/21270595

## Problem and motivation
Eye banks assess donor corneas with specular microscopy; manual workflows (the site names KONAN as an example device) are slow and variable. The winning model is to be piloted in eye banks.

## Data
- 9,000 training and 1,000 hidden test images (BMP), eye banks in NY, GA, KY, DE.
- The Zenodo record lists two related works: an AAO 2025 poster (PO200-2025, "A Proof-of-Concept Base Model for Automated Corneal Endothelial Cell Analysis via Specular Microscopy Images", presented by Almiqdad Saeed) and an IOVS/ARVO abstract ("An Automated Approach to Corneal Endothelial Graft Analysis via Specular Microscopy Images", Kalen Truong). Neither abstract text was found.

## Method
Baseline (GitHub README): Cellpose v1.0 'cyto' segmentation of the whole image; keep cells whose centroid falls in a deterministic per-image random crop of 40 % height x 40 % width; per-cell area, perimeter, neighbours/sides, centroid; CD = N x 1e6 / total area (um2) with 1000/1296 um/px; CV = std/mean area; HEX = mean IoU with a fitted hexagon.

## Results
- Metric and phases: see key_numbers (verified-from-source).
- Baseline scores: not reported publicly.
- Leaderboard numbers are in the team's BRIEF, not on the public pages read here.

## Limitations
- Reviewer: label procedure undocumented; organisers' baseline HEX definition differs from the stated metric definition ("six-sided cells / total cells").

## Relevance to CD / CV / HEX
- All three: the label generator is a black box; the only device hint is Konan. Konan's own patent ([[takahashi1996konanpatent]]) and eye-bank studies ([[vidalvillegas2024eyebank]], [[benetz2024donorecd]]) are the best available proxies.
- The site's HEX definition (six-sided / total) matches our finding that HEX equals the six-neighbour share of markers within 1-2 %.

## What it suggests we try
Treat "label = Konan software on technician-dotted cells in a technician-chosen region" as the working model and test it directly with a centre-method emulator on the 25 marker frames.

## Related notes
- [[takahashi1996konanpatent]] — Konan centre-method geometry.
- [[benetz2024donorecd]] — US eye-bank donor images with Konan CellChek D.
- [[pachitariu2025cellposesam]] — successor of the baseline's Cellpose.

## Sources
- https://clear-ec.grand-challenge.org/
- https://opticlab.med.upenn.edu/clear-ec-challenge
- https://zenodo.org/records/21270595
- https://github.com/CLEAR-EC/CLEAR-EC
