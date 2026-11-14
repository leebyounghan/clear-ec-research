---
id: find-packaging-bugs-outside-pod-tests
type: finding
name: 'Seven packaging bugs showed only on the real T4 and under uid 10001: every pod and Colab test ran as root inside the research environment'
summary: 'The staged package passed every pod test and still failed outside the pod three times: on the first real-T4 run every job answered with the fallback constants; after that fix a filtered timm file, unloading left off and a missing CV parameter folder appeared; and the first test as uid 10001 found a mode-600 code file, a mode-600 cpsam_v2 and a missing timm pin. A test that shares the research tree, its environment or root cannot see these.'
status: robust
metrics: []
date: '2026-11-13'
source: shared notes ledger_20261111.jsonl F-DEPLOY-002 (line 174), F-DEPLOY-003 (line 176), F-DEPLOY-005 (line 195), A-DEPLOY-008 / 009 / 010 / 011 (lines 175, 177, 186, 196), O-DEPLOY-T4-COLAB (line 184); docs/lane_summaries_20261113/final_summary_deploy_20261113.md sections 1 and 6; docs/FINAL_SUMMARY_20261113.md sections 1.3 and 5
evidence_for:
- '[[dec-020-t4-assumed-tier-p-t3-fallback]]'
---

# Seven packaging bugs showed only on the real T4 and under uid 10001: every pod and Colab test ran as root inside the research environment

The staged package passed every pod test and still failed outside the pod three times: on the first real-T4 run every job answered with the fallback constants; after that fix a filtered timm file, unloading left off and a missing CV parameter folder appeared; and the first test as uid 10001 found a mode-600 code file, a mode-600 cpsam_v2 and a missing timm pin. A test that shares the research tree, its environment or root cannot see these.

## Evidence

The seven bugs, with the test that found each and the fix:

1. PYTHONPATH not set in the image (real T4, Colab). The research pod runs with PYTHONPATH = `experiments/phase2_foundation_20261002/runtime` (dinov3 and a timm 1.0.20 copy); `cellpose.vit` imports the CPDINO backbones inside a bare try/except, so without the path every CPDINO member raised `NameError: name 'dinov3_vitb16' is not defined` and 13 of 13 jobs returned constants after reading 0.34 GB. Fix: Dockerfile ENV, `gc_entry.py` puts the runtime folder first on sys.path and raises at start-up if the backbones are missing.
2. `stage_package.py` dropped every file named `test_*.py`, which removed timm's own `runtime/timm/layers/test_time_pool.py`; ViT-L failed with ModuleNotFoundError (real T4). Fix: the runtime packages ship whole; the filter applies only to the lane's own inference tests.
3. `CLEAR_EC_UNLOAD` was set by every pod test but not by the image: all stages' models stayed on the GPU and the T4 job ran out of memory at the attention-pool stage (15.46 GB allocated) after 355 s (real T4). Fix: set in the Dockerfile and by default in `gc_entry.py`.
4. A missing CV parameter folder became cvp=None, a TypeError and constants (real T4, a test-layout artefact). Fix: `gc_entry.py` fails loudly at start-up when the CV parameters are missing.
5. `experiments/phase2_crossfold_20261015/source/phase2_lejepa_20261003.py` is mode 600 in the research tree; COPY kept it, every CD-family member raised PermissionError and the first p image (b982cbddf583) gave constants on 9 of 9 jobs (build box, uid 10001). Fix: `chmod -R a+rX` on the code tree and a build-time check as uid 10001.
6. `cpsam_models/cpsam_v2` is mode 600 in the staged model tree (submission 1's archive had it 644). Fix: the model archives are written with `--mode=a+rX`, owner 0:0.
7. The t2 / t3 requirements had no timm, but the Dockerfile asserts timm 1.0.20 at build time: the t3 build failed at step 4/9. Fix: `stage_package.py` pins timm 1.0.20 for every tier.

Reads: [[exp-deploy-colab-t4-timing-vm1]], [[exp-deploy-buildbox-container-test-p]], [[exp-deploy-buildbox-container-test-t3]]. Fixed contexts: A-DEPLOY-011 (Dockerfile sha256 2daeb1a20da917ce; context tree hash p 484c0d77e349502c, t3 d9e48922389c06ae).

## Limits

The platform itself has not run either image: upload and Try-out are the PI's actions ([[con-upload-and-submission-are-pi-actions]]). The uid-10001 test ran on an RTX 4090 with Grand Challenge's flags (--network none, 16 GB, read-only model), not on the platform. Every pod and Colab test (O-DEPLOY-015 / 017, O-DEPLOY-T4-COLAB, O-COORD-COLAB-T4-B) ran as root and covers no file permissions.

## Bears on

Lesson for any rebuild: test the image as its own user, with the extracted archive mounted read-only, no network and nothing of the research tree or its environment present; a pod test with the same PYTHONPATH and root proves nothing about the image. See also [[find-silent-fallback-constants-hide-failure]].
