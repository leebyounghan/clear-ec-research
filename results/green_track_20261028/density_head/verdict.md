# density_head — verdict (2026-10-28, fold-4 pilot)

Backlog item 1 closed as negative with data. A density head (frozen fold-4 recipe A tokens, strong point
supervision on the 20 train-green images, weak frame-integral supervision |log(integral/0.7528 mm^2) - log CD|
on all 7150 training rows, 3 epochs, 26 min) gives a frame-level CD estimate with r = 0.27 to log GT CD,
error correlation 0.13 with the adopted prediction and 52% MAPE after one scale; its density peaks localise no
cells on the 5 validation-green images (F1 0.0; in-ROI integrals 1.2-2x the marker counts). The uniform-density
weak label cannot be reconciled with the hidden-ROI labels: the head spreads density to satisfy the integral and
learns neither counts nor positions. Not extended (pre-registered rule: r_err <= 0.6 and MAPE <= 9).
