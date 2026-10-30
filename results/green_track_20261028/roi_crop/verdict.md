# roi_crop — verdict (2026-10-28)

Negative at both sizes. Re-predicting with the fold's adopted model on a crop around the detector cluster:
512x512 CD MAPE 8.2 / HEX 10.9 with error correlation 0.81-0.94 to the full-frame prediction, combined LOFO
blend +0.002 pp; 736x992 CD 7.3 / CV 11.5 / HEX 10.6 with correlation 0.90-0.97, +0.003 pp. The proposal
region adds nothing through the regressor (Phase 1's ROI result holds for this mechanism too). Side fact: the
model generalises to 512 crops better than expected (CD 8.2%).
