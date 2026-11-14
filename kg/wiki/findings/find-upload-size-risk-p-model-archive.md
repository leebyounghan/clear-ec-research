---
id: find-upload-size-risk-p-model-archive
type: finding
name: 'Unverified: the tier p model archive (15.06 GB) may exceed a 10 GB per-upload cap; the t3 archive (8.81 GB) is under it'
summary: The p model archive is 15.06 GB; submission 1's was 9.21 GB and the September model 8.0 GB, both accepted. Several Grand Challenge challenge pages state a per-upload cap of 10 GB for verified users, the official documentation states no model limit, and a platform admin wrote that the cap is not checked before the upload. Nobody has tried the upload.
status: tentative
metrics: []
date: '2026-11-13'
source: shared notes ledger_20261111.jsonl F-DEPLOY-006 (line 201), A-DEPLOY-014 (line 202), O-DEPLOY-013 (line 142); https://rnr-exm.grand-challenge.org/submission-qa/ ; https://grand-challenge.org/documentation/making-a-challenge-submission/ ; https://grand-challenge.org/forums/forum/the-leopard-challenge-719/topic/docker-image-limit-2615 (URLs as cited in the ledger line; not re-fetched by the KG ingest)
evidence_for:
- '[[dec-020-t4-assumed-tier-p-t3-fallback]]'
---

# Unverified: the tier p model archive (15.06 GB) may exceed a 10 GB per-upload cap; the t3 archive (8.81 GB) is under it

The p model archive is 15.06 GB; submission 1's was 9.21 GB and the September model 8.0 GB, both accepted. Several Grand Challenge challenge pages state a per-upload cap of 10 GB for verified users, the official documentation states no model limit, and a platform admin wrote that the cap is not checked before the upload. Nobody has tried the upload.

## Evidence

Model archives: p 15.06 GB (15,057,667,603 B), t3 8.81 GB (8,809,046,691 B); image archives 4.11 GB each. t2 (15.04 GB raw) and t1 (19.30 GB raw) are not under 10 GB either. The cap is stated as 10 GB for verified users and 2 GB for unverified ones on other challenges' pages; a refusal may come only after the full upload. Submission 1's package note records that the upload page states no size limit (O-DEPLOY-013).

## Limits

Unverified in both directions: no upload was attempted, and the two sources disagree (a cap on other challenges' pages against 'no size limit' on this challenge's upload page). The KG ingest did not re-read the web pages.

## Bears on

The Try-out note's rule: upload p as planned; if the model upload is refused or its import fails, switch both files to t3 ([[dec-020-t4-assumed-tier-p-t3-fallback]]). Bringing p under 10 GB per file is an untested idea: [[idea-p-model-archive-under-10gb]].
