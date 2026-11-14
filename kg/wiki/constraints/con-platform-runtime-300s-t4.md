---
id: con-platform-runtime-300s-t4
type: constraint
name: 'Platform limits: 300 s per case, 32 GB RAM, T4 assumed (16 GB)'
summary: 'The platform gives 5 minutes (300 s) per case, 32 GB RAM and a T4 (16 GB); whether an A10G can be selected is unconfirmed, and the PI said to assume the T4. Measured on Colab T4s with cold reads: tier p 287-288 s and 310.7-314.2 s, t3 173-174 s, t1 323-332 s; GPU peak 4.33 GB in every tier. The platform''s own T4 exec time is unmeasured.'
status: active
origin: literature
date: '2026-11-12'
source: 'docs/BRIEF.md (''Runtime: 5 minutes per case, 32 GB RAM, T4 or A10G''); results/submission1_package_20261031/PACKAGE.md and UPLOAD_TRYOUT_KO.md as read in shared notes ledger_20261111.jsonl O-DEPLOY-013 (line 142); D-PKG-002 (line 143), D-PKG-006 (line 170), O-DEPLOY-T4-COLAB (line 184), O-COORD-COLAB-T4-B (line 207), A-DEPLOY-014 (line 202); docs/FINAL_SUMMARY_20261113.md section 0'
constrains:
- '[[dec-019-package-tiers-20261112]]'
- '[[dec-020-t4-assumed-tier-p-t3-fallback]]'
- '[[method-deploy-tiers]]'
- '[[exp-deploy-tier-table]]'
- '[[idea-p-model-archive-under-10gb]]'
- '[[idea-cv-quartet-free-stack-shared-flips]]'
---

# Platform limits: 300 s per case, 32 GB RAM, T4 assumed (16 GB)

The platform gives 5 minutes (300 s) per case, 32 GB RAM and a T4 (16 GB); whether an A10G can be selected is unconfirmed, and the PI said to assume the T4. Measured on Colab T4s with cold reads: tier p 287-288 s and 310.7-314.2 s, t3 173-174 s, t1 323-332 s; GPU peak 4.33 GB in every tier. The platform's own T4 exec time is unmeasured.

## Rule

300 s per case (Try-out limit 300 s), 32 GB RAM, GPU NVIDIA T4 16 GiB as submission 1's active image shows; A10G (24 GB) unconfirmed as selectable. The upload page states no size limit; a 10 GB per-upload cap is possible ([[find-upload-size-risk-p-model-archive]]). Origin: the platform's limits as recorded in the project's own documents, and the PI's instruction to assume the T4 (D-PKG-006).

## Measured

| Tier | Colab T4 wall s per job, cold reads | H200 wall s | Strict mean |
|---|---|---|---|
| p | 287 / 288 (VM 1); 314.2 / 311.1 / 310.7 (VM 2) | 69.9 | 8.6682 |
| t3 | 173 / 174 (VM 1) | 36.0 | 8.6861 |
| t2 | 273 / 277 (VM 1) | 59.0 | 8.6728 |
| t1 | 323 / 323 / 332 (VM 1) | 70.9 | 8.6662 |
| full | not run on a T4 | 86.7 | 8.6679 |

GPU peak 4.33 GB on the device in every tier with unloading; RAM 3.0-5.7 GB. On the build box's RTX 4090 the p image's T4 path takes 72-80 s per job. Sources: [[exp-deploy-colab-t4-timing-vm1]], [[exp-deploy-colab-t4-timing-vm2]], [[exp-deploy-tier-table]], [[exp-deploy-buildbox-container-test-p]]. The only platform-measured time on record is Phase-1 alone, 44 s on the T4 (2026-09-14); submission 1's platform time was not recorded.

## Not measured

The platform T4 exec time of tier p (and of t3). Colab's VMs have 12 GB RAM, so every job there is a cold disk read; platform container start and archive mount are not included in any number above.

## How to apply

The Try-out exec time decides. Rule in the UPLOAD_TRYOUT note: p above 270 s exec time, or a refused model upload, means switch image and model to t3 ([[dec-020-t4-assumed-tier-p-t3-fallback]]). This is the one runtime check on the package track that [[con-no-runtime-caveats]] leaves room for; it is not a lane rule.
