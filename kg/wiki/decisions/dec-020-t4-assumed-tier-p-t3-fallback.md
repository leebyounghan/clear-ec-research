---
id: dec-020-t4-assumed-tier-p-t3-fallback
type: decision
name: 'T4 assumed: fp16 path as the container default, tier p built first and t3 as the fallback (D-PKG-004 to D-PKG-007)'
summary: 'The PI saw no GPU-type choice on the platform and said to proceed as if the GPU is the T4. The T4-safe path (fp16 + one detector copy below compute capability 8.0) became the build target. On a real Colab T4 tier t1 took 323-332 s and tier p 287-288 s (a second VM: 310.7-314.2 s) against the 300 s limit, t3 173-174 s. Tier p was built and tested as the image to upload first and t3 as the fallback; the Try-out note switches both files to t3 if p''s exec time is above 270 s or its model upload is refused. The upload and the final pick are the PI''s.'
status: accepted
date: '2026-11-12'
deciders:
- PI
- coordinator
source: shared notes ledger_20261111.jsonl D-PKG-004 (line 166), D-PKG-005 (line 169), D-PKG-006 (line 170), D-PKG-007 (line 193), F-DEPLOY-004 (line 187), O-COORD-BUILD-P (line 192), A-DEPLOY-012 / 013 / 014 (lines 197, 199, 202); deploy_UPLOAD_TRYOUT_KO_20261113.md; docs/FINAL_SUMMARY_20261113.md sections 0, 1.3 and 5
decides:
- '[[method-t4-fp16-one-detector-copy]]'
- '[[exp-deploy-buildbox-container-test-p]]'
- '[[exp-deploy-buildbox-container-test-t3]]'
- '[[exp-deploy-t4-default-200-rows-p]]'
alternative:
- '[[exp-deploy-t4-all-fp32-200-rows]]'
motivated_by:
- '[[exp-deploy-colab-t4-timing-vm1]]'
- '[[exp-deploy-colab-t4-timing-vm2]]'
- '[[exp-deploy-t4-precision-paths-20-frames]]'
- '[[con-platform-runtime-300s-t4]]'
depends_on:
- '[[dec-019-package-tiers-20261112]]'
---

# T4 assumed: fp16 path as the container default, tier p built first and t3 as the fallback (D-PKG-004 to D-PKG-007)

The PI saw no GPU-type choice on the platform and said to proceed as if the GPU is the T4. The T4-safe path (fp16 + one detector copy below compute capability 8.0) became the build target. On a real Colab T4 tier t1 took 323-332 s and tier p 287-288 s (a second VM: 310.7-314.2 s) against the 300 s limit, t3 173-174 s. Tier p was built and tested as the image to upload first and t3 as the fallback; the Try-out note switches both files to t3 if p's exec time is above 270 s or its model upload is refused. The upload and the final pick are the PI's.

## Context

D-PKG-003 had assumed an A10G (bf16). Submission 1's active image shows 'Requires GPU Type: NVIDIA T4, 16 GiB', and whether the A10G can be selected is unconfirmed. Per-stage unloading had already brought the GPU peak from 17.35 GB to 4.33 GB in every tier, so memory no longer rules the T4 out; time does ([[con-platform-runtime-300s-t4]]).

## Decision

- D-PKG-004 (coordinator, 11-12 18:40 EST): prepare a T4-safe dtype path so the answer does not depend on the A10G being selectable; measure parity, GPU peak and time on tier t1.
- D-PKG-005 (coordinator; stamped 21:10 EST): the PI asked for the container to be built for the A10G; primary tier t1. The move of the package to the build box was denied by the coordinator session's permission system and stays with the PI to run; nobody works around it.
- D-PKG-006 (PI; stamped 20:25 EST, and recorded as superseding D-PKG-005): proceed on the assumption that the platform GPU is the T4; the T4-safe path is the primary build target, tier t1 first; if t1 cannot fit 300 s on a T4 with parity, give the largest tier that does.
- F-DEPLOY-004 (deploy lane's recommendation, 11-12 23:55 EST): p first. It is the largest tier that finished inside 300 s on the real T4 with cold reads; t1 took 323-332 s; t2 buys only about 12 s over p for +0.005 strict; t3 (about 174 s, strict 8.6861) is the fallback if p times out.
- D-PKG-007 (coordinator on the PI's request, 11-13 20:12 EST): look at fp32 once more on 200 rows; the p package build is not held for it.

## Consequences

- Container default below compute capability 8.0: fp16 + one detector copy ([[method-t4-fp16-one-detector-copy]]); strict-score impact +0.0066 / +0.0067 on 200 labelled rows ([[exp-deploy-t4-default-200-rows-t1]], [[exp-deploy-t4-default-200-rows-p]]). All-fp32 reads +0.0069 and did not finish in 600 s on the real T4 ([[exp-deploy-t4-all-fp32-200-rows]], [[find-kernel-change-not-precision-moves-outputs]]).
- Real-T4 time: [[exp-deploy-colab-t4-timing-vm1]], [[exp-deploy-colab-t4-timing-vm2]].
- Built and tested on the build box as Grand Challenge runs it, 17 of 17 jobs each: tier p image dfc08a448e58 ([[exp-deploy-buildbox-container-test-p]]) and tier t3 image 8853d77c1f5a ([[exp-deploy-buildbox-container-test-t3]]). Seven packaging bugs were found and fixed on the way ([[find-packaging-bugs-outside-pod-tests]]).
- UPLOAD_TRYOUT note for the PI (A-DEPLOY-014): platform settings (GPU NVIDIA T4, memory 32 GB as in submission 1), expected Try-out outputs for debug cases 0001-19 / 20 / 21 for p and t3 (tolerance CD 0.1 %, CV 0.5 %, HEX 0.5 %), log checks (mode recipe, errors {}, t4_default true, autocast fp16, det_copies 1), and the rule 'Try-out exec time above 270 s on p -> switch to t3 (image and model together)'; a refused model upload also means t3 ([[find-upload-size-risk-p-model-archive]]).
- Superseded statements: 'T4 not viable' (O-DEPLOY-013) and 'A10G required' (D-PKG-003).

## Revisit when

The platform Try-out gives the exec time of p (unmeasured at close-out), or the model upload is accepted or refused. Both are the PI's actions ([[con-upload-and-submission-are-pi-actions]]). Unverified at close-out: `cellroute_deploy_{t2,t3}.json` against a no-outer-fold refit (the consolidated summary as revised at 22:19 EST 11-13: t2 and t3 still unverified; the p file's kept-version fit was recomputed on 11-13 by the post-processing analysis and reproduces to 2.8e-17, its version choice was not rerun).
