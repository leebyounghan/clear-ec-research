# Green day 2 — ranked summary (2026-11-01 20:00 UTC to 2026-11-02 ~02:30 UTC)

**Directive (PI, 2026-11-01):** use all four pod GPUs until about tomorrow to discover new ways to use green (the
cell-centre annotations / detectors). **Adoption rule (parent, 21:05 UTC):** increments under ~0.02 pp mean are
evidence-only; nothing below is proposed for submission 2. Base = hybrid B (8.8772 on 8954 rows; 8.8682 on the
8934 non-leaked rows), active hex6 = nine fold-99 detectors + pre-only s46, -0.0311 pp mean on B. Every number is
fold-clean, LOFO-calibrated, with the 20 train-green rows excluded. Protocol and gates: `protocol.json`.

## Ranked findings

| rank | finding | effect (mean pp on B) | gate | adoption |
|---|---|---|---|---|
| 1 | **Coverage-conditional hex6 weight (D2b):** blend weight per framed x detection-count tercile instead of a fixed 0.28 | -0.0374 vs -0.0311 (gain -0.0062, 5/5 folds, fold-4 HEX -0.057) | PASSED (>= 0.005) | evidence-only (< 0.02); deployable in principle as a 6-entry w lookup |
| 2 | **Distilled hex6 cannot replace the detectors (G3c):** regressor head trained to predict the ten-family hex6 reaches r 0.855 but blends at w = 0 | +0.0001 | FAIL (<= -0.02) | closes the "drop the detector passes" route; the hex6 gain is information orthogonal to the regressor |
| 3 | **Detector-based CD is capped by the oracle (D1):** expert-point CD on the 25 annotated images is 7.24% MAPE vs the device CD (B 7.66% there); every detector CD estimator has w = 0 on every fold and stratum, including quality-gated (D1b) and TTA/frame-aug detectors | 0.0000 | stop rule fired | CD route closed for good; no recall work can beat the regressor |
| 4 | **More full-frame members add little (G1/G2b):** ten + three TTA pre-only -0.0337 (gain -0.0026), ten + five -0.0334; TTA does not improve a single member (s46 -0.0082 vs plain -0.0107) | -0.0023 .. -0.0026 | FAIL (< 0.005) | evidence-only; member-count curve saturates at ~-0.034 |
| 5 | **Frame-augmented pre-only detector (G2):** finds more cells (840-1030 vs 660 per image) but is a weaker hex6 member | swap -0.0293, add -0.0315 | FAIL | closed |
| 6 | **Multi-task hex6-auxiliary fine-tune of recipe A (G3a/b):** aux = baseline (HEX 10.2776 vs 10.2778 on folds 0-3), control slightly worse (10.3007); B-level combined identical for both arms (-0.0019 vs B + hex6) | -0.0019 (both arms) | FAIL | closed; lambda-3 dose-response (G3d) below |
| 7 | Tissue-interior / cluster-window hex6 (D2), tissue-filtered deployed hex6 (D2d) | identical to control / framed-only gain with unframed loss | FAIL | closed |

### G3d — lambda 3 dose-response: null confirmed, wrong direction

HEX folds 0-3 pooled: lambda 3 10.3292 vs control 10.3007 vs baseline 10.2778 (lambda 1: 10.2776); fold 4 10.161 vs 10.140;
B-level combined +0.0080 vs B + hex6 (0/5 folds); distilled hex6 still w ~0 (+0.0002, r 0.853). Stronger auxiliary weight
hurts the main objective and adds no information to the distilled feature.

### Seeds 49/50 member curve: saturated

ten + tta_s49 -0.0334, + tta_s50 -0.0324, + all five TTA pre-only members -0.0334 (gain -0.0023), nine + five -0.0337.
The curve is flat from three independent members on; the best single new member (s49) is -0.0123 alone. Gate fails.

## What the day established

1. The only part of green that moves the official score is hex6 as a *blend feature*, and its gain scales with how
   many cells the detectors found: w should be ~0.15 where detections are sparse and ~0.45-0.50 where they are dense
   (framed images with few detections: HEX gain -0.035; framed with many: -0.248). This is the one new, deployable
   refinement of the shipped component, held back only by the 0.02 adoption bar.
2. The hex6 information is not recoverable from the regressor itself (G3c), so the ten detector passes are the price
   of the -0.031; the T4 numerics check (earlier today) showed the passes survive fp16.
3. CD cannot be improved from cell centres: the measurement itself (expert points vs device CD) disagrees by 7%,
   above the regressor's 6.3%, and detector estimators are 9-13% with error correlation 0.6-0.7 to B.
4. Framed images remain the weak spot for CD, but no green feature touches it (w = 0 in the framed stratum for every
   estimator, with or without quality gating).

## GPU use

20:10-02:10 UTC, all four GPUs through the parent's lane01/lane23 runners: G1 TTA (15 items), G2 frame-aug
(5 folds), G3 aux + control (10 runs x 50 min), G3d lambda 3 (5 runs), pre-only seeds 49/50 (10 pretrain + TTA items).
CPU passes with 4-worker pools after the PI's note.

## Files

`protocol.json` (all gates, pre-registered before results), `d1_cd_features/verdict.md`, `d2b_active_gate/verdict.md`,
`d2c_members/verdict.md`, `g3_hex6aux/verdict.md`, per-experiment `result*.json`; sources in `source/`; ledger
`experiments/green_track_20261028/ledger.json`.
