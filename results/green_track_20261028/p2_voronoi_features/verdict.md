# p2_voronoi_features — verdict (2026-10-28)

First positive signal on the green track. From the existing stride-8 specialist maps (threshold 0.35,
centroid-refined peaks, complete bounded Voronoi cells), the six-neighbour fraction hex6 improves HEX by
-0.044 pp (-0.015 pp on the mean) with LOFO blend weights 0.14..0.16 and improvement on 5/5 folds (calibrated
hex6 MAPE 11.5%, error correlation 0.84). hex6_top (best 0.052 mm^2 window) gives -0.039, 5/5. CV features
give nothing (cv_vor r_err 0.78, w 0), consistent with the oracle degradation (CV needs near-perfect points).
cd_vor reaches CD MAPE 8.67% but no blend gain (r_err 0.74). Below the gate; pre-registered confirmation is
the stride-2 detector (sg2_specialist, queued): HEX<-hex6 must improve on >= 4 folds there.
