---
id: dec-016-freeze-candidates-20261111
type: decision
name: Freeze candidates 2026-11-11 18:00 EST (D-CD-001, D-CV-001, D-HEX-004, D-HEX-005)
summary: 'CD: P1 x SCALE / recipe A s46+s47 snapshot / DINOv3-B seed mean TTA / ViT-L s46+s47 + mask density + stretch (6.1741 as computed, 6.1743 strict). CV: 6-component stack without order-loss DINOv3 (10.1101 / 10.1160). HEX: 39-column joint nested ridge (strict 9.71977) with the cell-route step in its kept version (9.70581 as computed, 9.71342 end to end strict). Integrated strict mean 8.66789 against the best pipeline 8.73627.'
status: accepted
date: '2026-11-11'
deciders:
- coordinator
source: shared notes ledger_20261111.jsonl D-CD-001, D-CV-001, D-HEX-004, D-HEX-005, O-VER-003, O-VER-004; docs/BRIEF.md 'Freeze 2026-11-11 18:00 EST'
decides:
- '[[exp-cd-current-stack]]'
- '[[exp-cv-lane-best-stack]]'
- '[[exp-hex-joint-nested-ridge]]'
- '[[exp-hex-cellroute-konan-step]]'
alternative:
- '[[exp-hex-seed-audit-chain]]'
- '[[exp-cv-order-loss-dinov3]]'
depends_on:
- '[[dec-012-freeze-pick-by-five-fold]]'
---

# Freeze candidates 2026-11-11 18:00 EST (D-CD-001, D-CV-001, D-HEX-004, D-HEX-005)

CD: P1 x SCALE / recipe A s46+s47 snapshot / DINOv3-B seed mean TTA / ViT-L s46+s47 + mask density + stretch (6.1741 as computed, 6.1743 strict). CV: 6-component stack without order-loss DINOv3 (10.1101 / 10.1160). HEX: 39-column joint nested ridge (strict 9.71977) with the cell-route step in its kept version (9.70581 as computed, 9.71342 end to end strict). Integrated strict mean 8.66789 against the best pipeline 8.73627.

## Decision

See [[exp-ref-freeze-candidate-integrated]]. HEX fallbacks: the 38-column ridge, then the 10-step chain. D-HEX-004 supersedes D-HEX-001 (chain), D-HEX-002 (38-column ridge) and D-HEX-003 (reverted chain).

## Consequences

Open for the PI: the challenge's container / model size and per-job time limits (Q-001), docker build approval, use of the second submission ([[dec-011-submission2-hold]]).

## Revisit when

The synthetic-pretraining follow-ups landed and were closed without changing the freeze ([[dec-017-synthetic-pretraining-followup]]).

## Update 2026-11-13

The freeze stood through close-out (PI, D-SYN-006: 'D-CD-001 stands'). Of the items left open for the PI: the platform limits were found in the project's records (300 s, 32 GB, T4 or A10G; [[con-platform-runtime-300s-t4]]); the docker build was approved on 11-12 ([[dec-019-package-tiers-20261112]]); what is built for upload is tier p with t3 as the fallback ([[dec-020-t4-assumed-tier-p-t3-fallback]]); submission 2 is still unused. The CD reads ordered after the 11-11 leaderboard changed nothing ([[dec-018-cd-hypotheses-after-leaderboard]]).
