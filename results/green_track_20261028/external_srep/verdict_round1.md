# external_srep — round 1 verdict (2026-10-30)

No change from pre-training as run (stride-2 head, 2000 updates on 307 srep images at the nominal 1.30x
scale, then the standard 20-image fine-tune, per fold from the fold's encoder):
- val-5 F1 0.650 / 0.636 / 0.643 / 0.671 / 0.609 (mean 0.642; the 20-image detectors: 0.62-0.69).
- whole-frame coverage on the validation rows identical to the 20-image stride-2 detector: hull of peaks
  0.109 vs 0.110 of the frame (median), fraction of 0.052 mm^2 windows with >= 20 peaks 0.375 vs 0.376;
  the detector still fires on one ROI-sized cluster.
- hex6 blend: single -0.056 pp HEX on A; as a 10th ensemble member B -0.0286 pp mean (per fold -0.042 /
  -0.028 / -0.038 / -0.026 / -0.010), unchanged from the 9-set.
- pre-training loss plateaued at 0.43 (the 20-image fine-tune reaches 0.30), so either the srep points did
  not transfer (device / scale gap) or 2000 updates were too few. Round 2 (protocol_round2.json) separates
  these: transfer check without fine-tune, 6000 updates, spacing-matched scale.
