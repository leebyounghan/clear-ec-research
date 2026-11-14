---
id: dec-019-package-tiers-20261112
type: decision
name: Package the frozen candidate now, with reduced tiers; tier p planned as the primary image (D-PKG-001, D-PKG-002, D-PKG-003)
summary: 'PI, 2026-11-12 13:45 EST: package D-CD-001 + D-CV-001 + D-HEX-004 + D-HEX-005 into a docker image now, with the lossless savings; build and local test only, no registry push, no upload. Because the full candidate read about 30 GB of weights per job and peaked at 17.35 GB of GPU memory, the coordinator had the reduced tiers t1 / t2 / t3 refit the same day and planned tier p (CD t1 + CV t1 + HEX t2 + cell-route on t2; strict mean 8.6682 against 8.6679 for the full candidate) as the primary image.'
status: accepted
date: '2026-11-12'
deciders:
- PI
- coordinator
source: shared notes ledger_20261111.jsonl D-PKG-001 (line 137), D-PKG-002 (line 143), D-PKG-003 (line 159), O-DEPLOY-013 (line 142), A-DEPLOY-007 (line 162); docs/FINAL_SUMMARY_20261113.md sections 0 and 1
decides:
- '[[method-deploy-tiers]]'
- '[[method-packaged-deploy-entry]]'
- '[[exp-deploy-tier-table]]'
- '[[exp-cd-deploy-tiers]]'
- '[[exp-cv-deploy-tiers]]'
- '[[exp-hex-deploy-tiers]]'
- '[[exp-hex-cellroute-step-on-tiers]]'
- '[[con-upload-and-submission-are-pi-actions]]'
supersedes:
- '[[dec-007-start-package-track]]'
motivated_by:
- '[[exp-ref-deploy-size-runtime]]'
- '[[exp-deploy-packaged-mode-parity]]'
- '[[con-platform-runtime-300s-t4]]'
depends_on:
- '[[dec-016-freeze-candidates-20261111]]'
- '[[dec-006-package-after-freeze]]'
---

# Package the frozen candidate now, with reduced tiers; tier p planned as the primary image (D-PKG-001, D-PKG-002, D-PKG-003)

PI, 2026-11-12 13:45 EST: package D-CD-001 + D-CV-001 + D-HEX-004 + D-HEX-005 into a docker image now, with the lossless savings; build and local test only, no registry push, no upload. Because the full candidate read about 30 GB of weights per job and peaked at 17.35 GB of GPU memory, the coordinator had the reduced tiers t1 / t2 / t3 refit the same day and planned tier p (CD t1 + CV t1 + HEX t2 + cell-route on t2; strict mean 8.6682 against 8.6679 for the full candidate) as the primary image.

## Context

The package track had waited for the freeze ([[dec-006-package-after-freeze]]); when to start it was an open PI decision ([[dec-007-start-package-track]]), which D-PKG-001 settles. The platform limits were found in the project's own records (O-DEPLOY-013): 300 s per case, 32 GB RAM, T4 (16 GB) or A10G (24 GB), no stated size limit, accepted so far an image of 4.05 GB gz and a model archive of 9.21 GB gz ([[con-platform-runtime-300s-t4]]).

## Decision

- D-PKG-001 (PI): package the final candidate with the lossless savings (fp16 copies of the regression members, base weights dropped, duplicate cpsam_v2 removed). Owner: deploy lane. Build and local test only: no push to any registry and no upload to Grand Challenge; no 'latest' tag ([[con-upload-and-submission-are-pi-actions]]).
- D-PKG-002 (coordinator): reduced tiers must be buildable the same day. CPU-only refits, strict five folds, incumbent in the same run, deploy parameters by the nested procedure with no outer fold (L5), written next to the freeze files, which stay untouched. CD: t1 = ViT-L final checkpoint only + recipe A e20 TTA only; t2 = no ViT-L; t3 = P1 + recipe A e20 TTA + mask only. CV: plain DINOv3 one seed. HEX: t2 = ridge without the mtcnn5 column; t3 = also without the attention-pool and cvw3 columns and the detector consensus if cheap. Deploy lane: unload models between stages.
- D-PKG-003 (coordinator, pending the PI's choice at upload): build tier p = CD t1 + CV t1 + HEX t2 + cell-route (t2) as the primary image and the full candidate as the second image.

## Consequences

Tier values, strict five folds, L5 passed, verifier checked CD, CV and HEX: [[exp-cd-deploy-tiers]], [[exp-cv-deploy-tiers]], [[exp-hex-deploy-tiers]], [[exp-hex-cellroute-step-on-tiers]], [[exp-ref-verifier-cv-tiers]], [[exp-ref-verifier-hex-tiers]]; integrated means full 8.6679, p 8.6682, t1 8.6662, t2 8.6728, t3 8.6861 ([[exp-deploy-tier-table]]). Tier p has about half the weights and forwards of the full candidate. The full candidate's extra CD weights buy nothing measurable ([[find-cd-full-extra-weights-nothing-over-t1]]).

D-PKG-003's statement that an A10G is required for both images (bf16) was superseded the same evening: [[dec-020-t4-assumed-tier-p-t3-fallback]].

## Revisit when

The PI chooses at upload. The freeze decision itself is unchanged ([[dec-016-freeze-candidates-20261111]]).
