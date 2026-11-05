# g1a_gt_select — verdict (2026-10-30, 400 scoring rows)

Negative as a selection signal. Across 8 checkpoints per fold the GT score r(log hex6, log GT HEX) on 400
fold-training rows moves within 0.38-0.56 with no trend and does not track held-out localisation: fold 4
selects step 3500 (r 0.450, val-5 F1 0.646) while step 1500 has F1 0.684; fold 2 selects step 500 (F1 0.611)
while step 3500 has 0.658. The selected detectors are ordinary members (single hex6 blend -0.056 pp HEX on A);
adding them makes a 14-detector ensemble with B -0.0282 pp mean (2/5 folds <= -0.03), unchanged. With 400
rows the standard error of r is ~0.045, the size of the whole observed range, so the score is noise-limited;
a 2000-row re-test on fold 4 is queued before closing the hypothesis.
