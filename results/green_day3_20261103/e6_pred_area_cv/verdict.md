# E6 — CV from E2-predicted areas at precision-oriented points (2026-11-03, 16:40 UTC): FAIL (w = 0)

Point sets in the cluster window: CPSAM v2 centroids (median 205 per image; FP 0.07 on the 25) and the consensus of
CPSAM centroids with a detector point within 0.5 x spacing (median 201; the pre-only detections are dense, so the
consensus filter removes only ~2 %). E2 CNN area predictions at those points (16 s + 33 s per GPU half).

| feature | LOFO on B, CV (mean pp) | w | est. CV MAPE | r(err, B err) | rows |
|---|---|---|---|---|---|
| CPSAM points, 10 % trimmed CV of predicted areas | 0.0000 | 0 | 13.58 | 0.80 | 8403 |
| CPSAM points, plain CV | 0.0000 | 0 | 13.50 | 0.80 | 8403 |
| consensus points, trimmed / plain | 0.0000 / 0.0000 | 0 | 13.39 / 13.27 | 0.79 | 8319 |

The predicted-area CV is no better than the geometric Voronoi CV (13.3-13.6 % vs 13-14 %), with the same 0.8 error
correlation to B, so nothing blends. Root cause flagged before running: E2's area head passed only the srep validation
(r 0.93) and failed the CLEAR-EC validation-green sanity (r 0.45 on 5 images, six-sided accuracy 0.65), i.e. the
appearance-to-area mapping learned on Topcon images does not transfer to specular CLEAR-EC frames. Closed.

Files: `e6_pred_area_cv/{result.json, features.npz, points_*.npz, pred_*.npz}`, `source/green_day3_e6_pred_area_cv_20261103.py`.
