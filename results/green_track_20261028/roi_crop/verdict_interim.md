# roi_crop — interim verdict (512x512 done, 736x992 running)

512x512 crops around the detector cluster: the adopted model predicts surprisingly well on crops (CD MAPE
8.2%, HEX 10.9%) but the errors are 0.81-0.94 correlated with the full-frame prediction; LOFO blend
+0.002 pp overall (2/5 folds), no gain. Consistent with Phase 1: the proposal region adds nothing through the
regressor. Final verdict after the 736x992 size.
