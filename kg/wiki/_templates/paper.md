---
id: firstauthorYEARkeyword          # BibTeX citekey, must equal the filename
type: paper
title: ""
authors: []                         # full list if short, else first 6 + "et al."
year: 2025
venue: ""                           # journal/conference; "arXiv" / "medRxiv" for preprints
doi: ""
arxiv: ""
pmcid: ""
url: ""
tier: core                          # core | candidate
group: ""                           # free text, e.g. endothelium-segmentation, specular-microscopy, cell-counting
status: reviewed                    # reviewed | to-read
review_depth: deep                  # deep | light
reviewed_on: 2026-11-08
sources_read: [full-text]           # full-text | html | pmc | abstract | repo | model-card
evidence: peer-reviewed             # peer-reviewed | preprint | workshop | review
tldr: ""                            # 1-2 sentences
relevance_circe: high               # field name kept from the EHR-KG template; here it means relevance to CLEAR-EC (high | medium | low)
relevance_note: ""                  # one line: why it matters (or not) for CD/CV/HEX from specular images
code_release: unknown               # open | partial | none | unknown
weights_release: unknown            # open | gated | none | unknown
data_scale: ""
model_scale: ""
key_numbers: []                     # short strings: "CD error 3.1% vs manual on 50 Topcon images"
results: []                         # rows: - {task, dataset, metric, value, comparator, comparator_value, split, setting, locator}
tags: []
search_keywords: []
# ---- typed edges (schema/edges.yaml). Ids that have no note become stubs + a lint warning, not an error ----
proposes: []                        # method-... / ds-...
uses_model: []
uses_dataset: []                    # ds-... (e.g. ds-topcon-external-385)
evaluates_on: []
uses_method: []                     # method-...
discusses: []
releases_code: []
extends: []
compares_with: []                   # - {target: "[[id]]", result: better, metric: "...", note: "...", locator: "Table 2"}
qualifies: []
studies: []
supports: []
disputes: []
critiques: []
same_lineage: []
has_limitation: []                  # - {target: "[[lim-...]]", stated_by: author|reviewer, note: "...", locator: "Sec 6"}
addresses: []
raises: []
evidence_for: []                    # CLEAR-EC: idea-... / dec-... / find-...  e.g. - {target: "[[idea-mixup-cutmix]]", note: "...", locator: "Table 3"}
evidence_against: []                # same targets
supports_finding: []                # CLEAR-EC: find-... this paper corroborates
---

# {Title}

## TL;DR

## Problem and motivation

## Data
Cohort, device, image size and scale, number of images, labels (manual markers, device values), split.

## Method

## Results
Numbers with locators.

## Ablations and analysis

## Limitations stated by the authors

## Limitations seen by the reviewer

## Reproducibility
Code, weights, data access, licence.

## Relevance to CLEAR-EC
What to take for CD, CV or HEX; which idea-, find- or method- nodes it bears on.

## Related work links
- [[id]] — relation in one line

## To read next

## Sources
