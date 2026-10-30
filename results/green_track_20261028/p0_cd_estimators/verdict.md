# p0_cd_estimators — verdict (2026-10-28)

Screen failed. Every grid-quantised point-map feature (peak counts, soft mass, window counts, grid spacing)
carries no CD information (|r| <= 0.30 with log GT CD; LOFO blend weight 0 on every fold). The detectors fire
on ~100 peaks per frame where ~2000 cells exist, and the 8-px grid cannot resolve cell spacing (29..21 px).
Centroid-refined peaks with a complete-6-neighbourhood spacing estimator do carry signal: specialist maps give
CD MAPE 8.91% after fold-clean log-linear calibration (slope 0.8), but the error correlation with recipe A is
0.72, so the LOFO blend picks w = 0..0.04 and the delta is +0.007 pp. The frozen probe (F1 0.47) gives 9.36%,
i.e. localisation quality moved this estimator by only 0.45 pp. Side result: an intercept-only log-space L2
calibration is +0.059 pp worse than the adopted MAPE-optimal scalar, which explains most of the earlier ridge
stacking's +0.07..0.10 pp — those features were "no signal", not harmful.
