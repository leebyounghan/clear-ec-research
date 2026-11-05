# external_srep — round 2 final (2026-10-30 22:30 UTC)

Confirmation of the pre-only mechanism with two more head seeds (pre-training only, no CLEAR-EC fine-tune):
- singles on A: pre-only s46 -0.064, s47 -0.069, s48 -0.050 pp HEX (fine-tuned detectors: -0.045..-0.069).
- nine + three pre-only (12 detectors), base B: -0.0978 pp HEX = -0.0326 pp mean, HEX improving 5/5, 4/5 folds
  <= -0.03 (per fold -0.0489 / -0.0374 / -0.0404 / -0.0306 / -0.0058); base A -0.160 pp HEX = -0.0533 (4/5).
  Full-fit on all rows: a -0.195, b 0.610, w 0.30 (deploy/hex6_blend_params_12.json).
- nine + one pre-only (s46), base B: -0.0311 (3/5 folds); full fit a -0.208, b 0.599, w 0.28.
Parent's fixed rule: -0.0326 lies in the -0.030..-0.035 band -> submission 2 = nine + the single best pre-only
(the best single is chosen by its own 10-detector LOFO number on B, computed next).
Fine-tuned external variants all land under the bar: spacing-matched -0.0294, 6000-update -0.0299, joint
-0.0274 (joint single -0.058): the 20-image fine-tune is what removes the external benefit.
Multiplicity caveat stands: the pre-only member was one of five pre-registered round-2 variants; the
confirmation seeds were queued after the first pass and improve the ensemble monotonically (10 -> -0.0311,
12 -> -0.0326), which is the expected behaviour of independent members rather than a lucky draw.

## Decision applied (2026-10-31 15:25 UTC)
Single pre-only members as 10th detector, LOFO on B: s46 -0.0311 (3/5 folds), s48 -0.0305, s47 -0.0303; all 5/5 HEX
improving. Submission 2 = nine + pre-only s46 (parent's middle band). Activated deploy/hex6_blend_params_10.json
(a -0.2084, b 0.5989, w 0.28, 8925 rows; LOFO -0.0311 pp mean on B, per fold -0.047 / -0.031 / -0.042 / -0.027 /
-0.009); the frozen nine-only parameters are kept as hex6_blend_params_frozen9.json. Tests v2 queued.
