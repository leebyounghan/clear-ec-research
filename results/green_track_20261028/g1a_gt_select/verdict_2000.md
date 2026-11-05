# g1a_gt_select — 2000-row re-test (2026-10-30), closes G1a

Negative, now with a properly powered score (SE of r ~0.02): on fold 4 the GT score moves within 0.47-0.51
across the 8 checkpoints and still does not track held-out localisation (selected step 2000, r 0.509, val-5 F1
0.651; step 1000 had F1 0.661). Image-level GT HEX cannot select point detectors by checkpoint; G1a is closed.
