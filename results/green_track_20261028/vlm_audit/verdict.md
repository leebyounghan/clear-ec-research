# vlm_audit — Claude-in-the-loop audit, round 1 (2026-10-28)

Loop: contact sheets of the 12 worst and 12 best HEX OOF residuals (adopted pipeline) with the fold-clean
specialist detections overlaid (green = complete bounded Voronoi cell, red = other peak), inspected by Claude.

Observations
1. The point specialist does not cover the frame. On every image it fires on one or two compact clusters of
   roughly 100-300 peaks in the best-focus region, i.e. it has learned the annotators' ROI-selection behaviour
   from the 20 ROI-masked training images. Its cluster is an ROI proposal, not a frame-wide cell map. This
   explains p0 (frame-level counts are uninformative) and why hex6 on the cluster carries HEX information.
2. The worst HEX residuals are pathological or near-blank corneas (CD 776-1408, guttae, CV up to 0.95): the
   regressor over-predicts HEX by +50..+70% (GT 0.20-0.30, predicted 0.37-0.64). Where the detector finds a
   cluster there, hex6 is low (0.13-0.24) and corrects in the right direction; where it finds < 20 peaks the
   feature is undefined and the blend falls back. The remaining worst cases have normal-looking mosaics with
   GT HEX 0.28-0.37, where hex6 agrees with the regressor; these look like label noise and are not fixable.
3. On the best residuals (normal mosaics) hex6, prediction and GT coincide, so the blend costs nothing there.

Derived hypotheses (coded and tested fold-clean, not tuned by eye)
- H1 (p2b, done): hex6 should include irregular cells; the completeness filter hurts -> variant F, -0.054 pp.
- H2 (roi_crop, queued): re-predict with the fold's recipe A model on a crop centred on the detected cluster
  and blend by LOFO; the hypothesis is that the cluster approximates the annotator's region better than the
  full frame and that its errors differ from full-frame std pooling's. Two crop sizes pre-registered.
- H3 (not pursued): a confidence gate on the number of bounded cells; the NaN fallback already covers the
  undefined cases, and the blend weights are small enough that noisy hex6 values cost little.
