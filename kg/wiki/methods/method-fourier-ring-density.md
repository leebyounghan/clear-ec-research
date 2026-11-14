---
id: method-fourier-ring-density
type: method
name: Label-free Fourier ring cell-density estimate
summary: 'Cell density from the radial power spectrum: Hann-windowed spectrum, radial mean, one quadratic 1/f fit in log-log over 0.006-0.25 cycles/px, band peak 500-5000 cells/mm2 with parabolic refinement, D = (sqrt(3)/2)(f0/s)^2 1e6; full frame and 70 tiles of 256 px. One fixed form, no label used.'
code: lane_cellroute_20261109/source/fft_spacing.py (md5 220d9929)
---

# Label-free Fourier ring cell-density estimate

Cell density from the radial power spectrum: Hann-windowed spectrum, radial mean, one quadratic 1/f fit in log-log over 0.006-0.25 cycles/px, band peak 500-5000 cells/mm2 with parabolic refinement, D = (sqrt(3)/2)(f0/s)^2 1e6; full frame and 70 tiles of 256 px. One fixed form, no label used.

## Where it is used

Experiments link here with `uses_method`; see backlinks. Sources: ledger D-CDSIMPLE-001 (a), A-DEPLOY-005; `lane_cellroute_20261109/source/fft_spacing.py`, `feats/fft_spacing_cd.npz`.
