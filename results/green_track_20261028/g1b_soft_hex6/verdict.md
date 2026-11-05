# g1b_soft_hex6 — verdict (2026-10-30, surrogate test)

The differentiable surrogate failed its pre-registered 20-image test and the weak-loss training was not run.
On the rasterised expert heatmaps the soft hex6 correlates -0.15 with the discrete hex6 of the same points
(discrete hex6 vs GT HEX: 0.84); on the detector heatmaps 0.28; autograd gradients differ from finite
differences by 2-4x at the checked cells. Diagnosis: with sigma-8 Gaussian blobs on a 2-px grid the soft-peak
mass of one cell spreads over several grid cells, so the annulus neighbour count is inflated unevenly and the
Gaussian-around-6 weighting collapses; the ratio-to-maxpool term is also non-smooth at ties. A faithful
surrogate would need per-blob normalisation (a soft argmax per connected blob), which is a research item of
its own; given the external-data lead and the deadline it is parked unless the PI asks for it.
