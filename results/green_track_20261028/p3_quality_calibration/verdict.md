# p3_quality_calibration — verdict (2026-10-28)

Negative. Letting the per-target calibration scalar depend on a detector-derived quality tercile changes the
mean by +0.002..+0.003 pp (2-3 folds improving by < 0.01); the global refit reproduces the adopted pipeline
(-0.0002). The regression's residual bias does not depend on these quality proxies.
