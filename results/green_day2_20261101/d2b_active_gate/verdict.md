# D2b — coverage-conditional blend weight for the ACTIVE hex6 (2026-11-01, 21:00 UTC)

Pre-registered 20:50 UTC. Control = plain ten-family LOFO on B, reproduced exactly: **-0.0311 mean pp** (HEX -0.0934, w 0.26/0.28/0.28/0.28/0.32).
Per-family hex6 and detection counts recomputed from the fold-clean bf16 OOF outputs with the deployed definition
(`d2b_active_gate/active_family_features.npz`). Strata are label-free and available at inference.

| strata (w chosen per stratum on the fitting folds) | mean pp | gain vs control | folds improving | dev HEX pp (f0-3) | fold-4 HEX pp | w (fold 0, low -> high) |
|---|---|---|---|---|---|---|
| summed detection count, terciles | -0.0354 | -0.0043 | 5 | -0.122 | -0.043 | 0.16 / 0.28 / 0.42 |
| summed detection count, quintiles | -0.0351 | -0.0040 | 5 | -0.117 | -0.057 | 0.16 / 0.16 / 0.14 / 0.50 / 0.38 |
| pre-only member count, terciles | -0.0346 | -0.0034 | 5 | -0.118 | -0.045 | 0.14 / 0.22 / 0.44 |
| inter-family spread of log hex6, terciles | -0.0334 | -0.0022 | 5 | -0.111 | -0.058 | 0.44 / 0.24 / 0.14 (low spread -> high w) |
| framed / unframed only | -0.0311 | +0.0001 | 5 | -0.109 | -0.029 | 0.32 / 0.20 |
| **framed x summed-count terciles (6 strata)** | **-0.0374** | **-0.0062** | 5 | -0.126 | -0.057 | 0.18 / 0.44 / 0.34 (unframed) ; 0.14 / 0.20 / 0.50 (framed) |

**Gate (>= 0.005 better than control, fold 4 negative, >= 4/5 folds): passed** by the framed x count variant; the plain
count terciles fall just short (-0.0043). Mechanism: the hex6 correction deserves more weight where the detectors
found many cells (high coverage = reliable hex6) and almost none on framed images with few detections
(HEX gain -0.035 in that stratum vs -0.248 for framed images with many detections).

**Adoption: evidence-only.** Parent rule (21:05 UTC, docs/IDEA_REGISTRY_20261031.md): increments under ~0.02 pp mean are
not adopted for submission 2 (0.005 is about one paired SE; container changes carry try-out risk). No deployment work.
Recorded as a documented, deployable-in-principle refinement (w lookup by black_frac and total detection count).

Files: `d2b_active_gate/{result.json, run.log, active_family_features.npz}`, `source/green_day2_d2b_active_gate_20261101.py`.
