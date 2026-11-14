---
id: find-silent-fallback-constants-hide-failure
type: finding
name: A per-image fallback to constants hid a broken pipeline four times (CD 2770, CV 0.36, HEX 0.53 on every job)
summary: 'The entry answers an image with the submission-1 constants when a member raises. Four environment faults therefore produced well-formed outputs on every job within seconds: the missing PYTHONPATH (13 of 13 Colab jobs), a missing CV parameter folder, a code root laid out as a symlink on the second Colab VM, and a mode-600 file under uid 10001 (9 of 9 jobs). Start-up checks now fail loudly instead.'
status: robust
metrics: []
date: '2026-11-13'
source: shared notes ledger_20261111.jsonl F-DEPLOY-002 (line 174), O-DEPLOY-T4-COLAB (line 184), O-COORD-COLAB-T4-B (line 207), F-DEPLOY-005 (line 195), A-DEPLOY-008 (line 175), A-DEPLOY-010 (line 186), A-DEPLOY-014 (line 202)
---

# A per-image fallback to constants hid a broken pipeline four times (CD 2770, CV 0.36, HEX 0.53 on every job)

The entry answers an image with the submission-1 constants when a member raises. Four environment faults therefore produced well-formed outputs on every job within seconds: the missing PYTHONPATH (13 of 13 Colab jobs), a missing CV parameter folder, a code root laid out as a symlink on the second Colab VM, and a mode-600 file under uid 10001 (9 of 9 jobs). Start-up checks now fail loudly instead.

## Evidence

- F-DEPLOY-002: every job answered CD 2770, CV 0.36, HEX 0.53 within 7-15 s, all modes; the entry mode in the output was 'constant' with the error recorded per image.
- O-DEPLOY-T4-COLAB: the t2 / t3 'constant' jobs of run6 came from a missing lane_cv deploy_t2 / deploy_t3 folder in the test layout.
- O-COORD-COLAB-T4-B: the coordinator's first run on the second VM returned constants in 10 s for every job (code root as a symlink).
- F-DEPLOY-005: the first p image gave 2770 / 0.36 / 0.53 on 9 of 9 jobs under uid 10001.

By design a garbage input file still returns the constants (6 s in the container test, [[exp-deploy-buildbox-container-test-p]]).

## Limits

The fallback itself is kept for unreadable inputs; what changed is that environment faults (missing backbones, missing CV parameters, unreadable code) now stop the job at start-up.

## Bears on

A constant output is a valid submission row, so a platform run can 'succeed' while predicting nothing. The Try-out checks in the UPLOAD_TRYOUT note (mode recipe, errors {}, t4_default true, autocast fp16, det_copies 1, and the expected outputs of debug cases 0001-19 / 20 / 21) exist for this reason. See [[find-packaging-bugs-outside-pod-tests]].
