# local_pool — verdict (2026-10-28)

Negative for the region hypothesis. Ridge heads in standardised-log space on frozen fold-encoder tokens
(alpha 1000 on every fold by inner CV): validation means 8.77-9.24 (global), always worse for the probe-weighted
(soft) and 512-box poolings. LOFO blend with the adopted pipeline: global control -0.019 pp (CD -0.029, CV
-0.010, HEX -0.018; error correlation 0.97-0.99), soft -0.016, box -0.010. The region variants never beat the
global control, so the detector's proposal region carries no information beyond global std pooling; the small
control gain is head diversity (a second head on the same tokens), not a green effect, and is below the gate.
