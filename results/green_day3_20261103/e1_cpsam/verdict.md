# E1 — Cellpose-SAM cells as an independent source (2026-11-03, 16:28 UTC): all gates FAIL

cpsam_v2 (foundation, no CLEAR-EC fine-tune) on all 8954 full frames, 0.6-0.8 s/image on an H200 (8877 images with
cells; 8613 with >= 12 cells after the window/border rules). Per-cell: centroid, mask area, mask-adjacency sides,
organiser hexagonality, border flag.

| feature | target | LOFO on B (mean pp) | w per fold | est. MAPE | r(err, B err) |
|---|---|---|---|---|---|
| hex6 on CPSAM centroids (deployed definition), alone | HEX | -0.0010 (4/5) | 0.06-0.10 | 11.07 | 0.89 |
| ten + hex6_cpsam (11th member) | HEX | -0.0317 vs control -0.0311 (gain -0.0006) | | | |
| swap sg2pre_s46 -> hex6_cpsam | HEX | -0.0302 (worse) | | | |
| six-neighbour fraction from mask adjacency (window / all) | HEX | 0.0000 (w = 0) | 0 | 11.1 | 0.91 |
| organiser hexagonality on masks (window / all) | HEX | 0.0000 (w = 0) | 0 | 12.1 / 11.9 | 0.86 |
| CV of mask areas (window / all, 5-95 % trimmed) | CV | 0.0000 (w = 0) | 0 | 13.1 / 13.2 | 0.83 |

Centroid accuracy vs the expert points inside the ROI box (12-px match): F1 0.520 on the 25 (0.446 on the five
validation-green images), far below the fine-tuned detectors (val-5 F1 ~0.64) and nowhere near the HEX target
from E4 (recall >= 92 %, localisation < 4 px). The CPSAM masks are not an accurate cell map of specular images, so
"independent" errors do not pay here: the hex6 member is the weakest of all tried (pre-only s46 alone -0.0107),
and mask-area CV / mask hexagonality carry nothing beyond B (error correlation 0.83-0.91).

Closed. CPSAM centroids and areas remain available for E5 (robust CV) as one more source, but the F1 says they will
not carry it. Files: `e1_cpsam/{result.json, features.npz, v2/rows_*.npz}`, `source/green_day3_e1_cpsam_20261103.py`, `source/green_day3_e1_evaluate_20261103.py`.
