# deploy (G3) — status (2026-10-30)

Submission-2 hex6 deliverable exists: nine detectors retrained from the full-data recipe A encoder
(val-5 F1 0.63-0.68, information only), full-fit blend parameters on base B (a = -0.223, b = 0.584, w = 0.26,
8921 rows; LOFO estimate -0.0274 pp mean), standalone deploy/hex6_correction.py (Hex6Corrector; 0.68 s per
image for nine detectors on an H200) and deploy/test_hex6_correction.py, all checks passed: deterministic,
CD/CV unchanged, finite, HEX ratio bounded, fallback on a blank image, feature available on all 25 annotated
images, r = 0.73 with the expert-point hex6 and 0.78 with GT HEX on those images. If the external pre-training
round 2 wins on the gates, the nine detectors are retrained with the pre-trained initialisation.
