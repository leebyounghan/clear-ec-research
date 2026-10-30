# p5_hex6_combo — verdict (2026-10-28 18:30 UTC)

GATE PASS, marginal. Variant A (pre-registered primary: geometric mean of hex6_all over the stride-8 specialist
and the stride-2 seed-46 head, no fitted combination parameter) blended by LOFO into the adopted pipeline:
HEX 10.253 -> 10.162 (-0.091 pp), mean 8.9757 -> 8.9453 (-0.0305 pp), CD/CV untouched, HEX improved on 5/5
folds; per-fold mean deltas -0.047 / -0.029 / -0.032 / -0.029 / -0.016 (two folds at or beyond -0.03).
Blend weights 0.22-0.26; calibrated hex6 MAPE 11.1%, error correlation 0.86. The two detectors' log hex6
correlate only 0.68, so averaging removes detection noise: single detectors give -0.054 and -0.063.
Variant B (count-gated weight on the stride-2 feature) adds nothing over the plain blend (-0.062).
Reported to the parent at 18:30 UTC with caveats (marginal; feature definition chosen on the stride-8 maps;
same 20 training images for both detectors; single-model baseline). Strengthening: seeds 47/48 and TTA of the
stride-2 head are queued; p6 recombines over all detectors when they exist.

## Addendum 19:05 UTC — hybrid base (parent request)

Against the submission frontrunner B = 0.5 recipe A (2-seed) + 0.5 Phase-1 five (8.8682 on the 8934 rows), the
same feature and protocol give -0.045 pp HEX = -0.015 pp mean, 5/5 folds improving, no fold at -0.03; the
two-seed recipe A base gives -0.032 (3 folds). Verdict: ABSORBED by the hybrid (the Phase-1 members already
carry about half of the hex6 information). The 4-detector ensemble (p6) will be reported on both bases; it is
not raised again unless it reaches -0.03 pp on B. Table: hybrid_base_result.json.
