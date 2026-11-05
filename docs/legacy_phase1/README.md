# Phase I legacy

Phase I is closed. Nothing here is a target, a yardstick or a live instruction. Current work starts at the [project state](../STATE_20261103.md).

## Comparison rule (PI decision, 2026-11-05)

Results are compared with our Phase II submissions only. Phase I scores, ours (8.7010) or anyone's, including the Phase I first place, do not enter a comparison, because they came from another test set of 100 images. The reference is Phase II submission 1: leaderboard 8.9675, OOF 8.877.

## Still in use, not legacy

The crop CPSAM members trained during Phase I are half of hybrid B, which is Phase II submission 1. `phase1_app/` inside `submission_hybrid_20261030/`, `submission_hybrid_hex6_20261101/` and `submission_final_stretch_20261104/` runs them, and the 18 modules in the root of [`code/`](../../code/README.md) are pinned by them. "Phase-1 members" in current documents names where those models came from.

## Moved here on 2026-11-05

These lived directly in `docs/`. Paths inside the published [technical report](../../output/pdf/Lamda_KIM_CLEAR_EC_Technical_Report.pdf) and in the frozen `docker_review_20260914/scripts/build_technical_report.py` still say `docs/<name>`: read them as `docs/legacy_phase1/<name>`.

| File | Title |
|---|---|
| [ALL_MEMBERS_SSL_20260915.md](ALL_MEMBERS_SSL_20260915.md) | Archived research record |
| [ARTIFACTS.md](ARTIFACTS.md) | September 14 release artifacts |
| [CPSAM_METHODS_20260914.md](CPSAM_METHODS_20260914.md) | Archived research record |
| [CROP_DINO_SCREEN_20260910.md](CROP_DINO_SCREEN_20260910.md) | September 10-11 — artifact-guard crop screen |
| [DEPLOY_METHODS_20260905.md](DEPLOY_METHODS_20260905.md) | September 5 — final fitting records |
| [DEPLOY_REVIEW_20260905.md](DEPLOY_REVIEW_20260905.md) | September 5 — initialization and inference audit |
| [DINO_ALL9000_20260913.md](DINO_ALL9000_20260913.md) | Archived research record |
| [DISTILLATION_NEXT_20260915.md](DISTILLATION_NEXT_20260915.md) | Archived research record |
| [DOCKER_SMOKE_20260915.md](DOCKER_SMOKE_20260915.md) | Released-container smoke test — September 15, 2026 |
| [EXTERNAL_DATA.md](EXTERNAL_DATA.md) | External data — historical research only |
| [FINAL_CANDIDATE_20260904.md](FINAL_CANDIDATE_20260904.md) | September 4 — all-data candidate |
| [FINDINGS.md](FINDINGS.md) | Development findings |
| [FOLLOWUP_20260902.md](FOLLOWUP_20260902.md) | September 2-3 — correctness audit |
| [LEADER_FOLLOWUP_20260911.md](LEADER_FOLLOWUP_20260911.md) | Archived research record |
| [METHODOLOGY.md](METHODOLOGY.md) | September 14 method |
| [METHODS_20260904.md](METHODS_20260904.md) | September 4 — training recipe comparison |
| [NEXT_STEPS.md](NEXT_STEPS.md) | Phase I technical-review checklist |
| [PACKAGING_20260909.md](PACKAGING_20260909.md) | September 9-10 — original image packaging |
| [PLATFORM_BINDING_20260915.json](PLATFORM_BINDING_20260915.json) | Verification record |
| [RAW_SSL_NEXT_20260913.md](RAW_SSL_NEXT_20260913.md) | September 13 — raw-label and SSL comparison |
| [REGION_SEARCH_20260905.md](REGION_SEARCH_20260905.md) | September 5 — native region selection |
| [REGION_SEEDS_20260906.md](REGION_SEEDS_20260906.md) | September 6-7 — region confirmation and integration |
| [REPRESENTATION_SSL_20260913.md](REPRESENTATION_SSL_20260913.md) | Archived research record |
| [REPRODUCE.md](REPRODUCE.md) | Reproduction scope |
| [SPECIALIZE_SCREEN_20260908.md](SPECIALIZE_SCREEN_20260908.md) | September 8 — metric specialization |
| [SPECIALIZE_SCREEN_20260908_audit.json](SPECIALIZE_SCREEN_20260908_audit.json) | Verification record |
| [SSL_DURATION_20260914.md](SSL_DURATION_20260914.md) | Archived research record |
| [SSL_RELEASE_20260914.md](SSL_RELEASE_20260914.md) | September 14 release — sixth-only SSL + raw labels |
| [SSL_SCREEN_20260911.md](SSL_SCREEN_20260911.md) | September 11 — native, fold-clean SSL screen |
| [STATRES_SCREEN_20260908.md](STATRES_SCREEN_20260908.md) | September 8 — residual feature statistics |
| [STATRES_SCREEN_20260908_audit.json](STATRES_SCREEN_20260908_audit.json) | Verification record |
| [STATUS_20260915.md](STATUS_20260915.md) | September 14 release — documentation entry point |
| [SUBMISSION_20260904.md](SUBMISSION_20260904.md) | September 4 — historical candidate preparation |
| [TARGET_REPLICATION_20260912.md](TARGET_REPLICATION_20260912.md) | September 12 — target policy replication |
| [TARGET_SCREEN_20260912.md](TARGET_SCREEN_20260912.md) | September 12 — target policy screen |
| [VENV_SMOKE_20260915.md](VENV_SMOKE_20260915.md) | Fresh source-venv inference verification |

## Legacy that stays where it is

Release material and research evidence are pinned by checksums, scripts or the published report, so they were labelled and not moved.

| Path | What it is |
|---|---|
| `docker_review_20260914/`, `training_reproduction/`, `review_submission_20260910/` | The September 14 release: Docker guide, training workspace, historical source |
| `submission/`, `submission_candidate_*/` | Phase I packages and candidates |
| `output/pdf/` | Phase I technical report |
| `code/archive/phase1_2026-09/` | 172 Phase I research scripts |
| `docs/archive/` | Phase I launch logs and superseded records |
| `results/` entries dated up to `20260923`, the undated `regress_*` and `ens_*` JSONs, and `results/leaderboard/` | Phase I evidence, listed in [`results/README.md`](../../results/README.md) |
| `results/recipe_A_20261027/leaderboard_phase1_20261030.md` | Phase I leaderboard snapshot |
