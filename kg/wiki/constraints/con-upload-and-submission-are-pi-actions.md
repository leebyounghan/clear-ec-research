---
id: con-upload-and-submission-are-pi-actions
type: constraint
name: 'Build and local test only: registry push, upload, Try-out and submission are the PI''s actions'
summary: 'PI, with the packaging approval of 2026-11-12: build and local test only. No push to any registry and no upload to Grand Challenge; the upload, the Try-out and the submission are PI account actions. No ''latest'' tag.'
status: active
origin: user
date: '2026-11-12'
scope: global
source: shared notes ledger_20261111.jsonl D-PKG-001 (line 137), D-PKG-005 (line 169), Q-DEPLOY-002 (line 141), Q-DEPLOY-003 (line 172), O-COORD-COLAB-T4 (line 171)
---

# Build and local test only: registry push, upload, Try-out and submission are the PI's actions

PI, with the packaging approval of 2026-11-12: build and local test only. No push to any registry and no upload to Grand Challenge; the upload, the Try-out and the submission are PI account actions. No 'latest' tag.

## Rule

Build and local test only: no push to any registry, no upload to Grand Challenge, no 'latest' tag. The upload, the Try-out and the submission are PI account actions (D-PKG-001).

## How to apply

Moves that a session's permission system denies stay with the PI to run; nobody works around them (D-PKG-005). That happened three times: the copy of the staged tree to the build box (Q-DEPLOY-002; after the PI's key registration the pod-to-box rsync ran at 07:30 EST 11-13), the copy of test files to the Colab runtime (Q-DEPLOY-003; the PI ran the copy script), and the move of package_20261112/t1 from the coordinator session. The PI's Colab login was placed on the pod for the T4 test and is to be removed after it (O-COORD-COLAB-T4); the deploy lane's summary lists it as still present at close-out.
