# E9 — label-verifier feasibility (2026-11-03, 23:20 UTC): clearly flat; route A (label-verified pseudo-labels) closed

Pre-registered in `protocol.json` before the run. Windows 538x408 at stride 16 (1728 per image), CD_est = k x peaks /
area with one global k (pre-only 1.70, compact 3.92), HEX_est = hex6 of bounded Voronoi cells centred in the window,
D = |log CD ratio| / s_CD + |log HEX ratio| / s_HEX (s_CD, s_HEX = label log-SDs). CPU pool of 3 workers, 245 s.

## Report 1 — the 25 annotated images (min-D window vs the expert ROI box)

| point source | subset | IoU median (mean) | centre distance median | rank of the expert-box window (percentile among valid windows) | D(expert box) - D(best) |
|---|---|---|---|---|---|
| pre-only s46 (full frame, clean on all 25) | validation-5 | 0.50 (0.47) | 120 px | 425 of ~1400 (30th pct) | +1.29 |
| pre-only s46 | train-20 | 0.35 (0.37) | 189 px | 595 (38th pct) | +1.69 |
| stride-2 s49 (compact; 20 leaked) | validation-5 | 0.25 (0.29) | 281 px | 1151 (79th pct) | +4.56 |
| stride-2 s49 | train-20 | 0.27 (0.29) | 237 px | 691 (68th pct) | +3.76 |

The 0.50 IoU of the full-frame source is not a verifier signal: the expert boxes sit near the frame centre (median
top-left 401 / 297 vs 379 / 282 for the centred window), so most central windows overlap them by ~0.5-0.8 regardless of
labels. The rank statistic is the test, and the expert-box window ranks like a random window (30-38th percentile);
with the compact source it ranks worse than random because the cluster is pulled elsewhere.

## Report 2 — all 8954 images: own labels vs 20 null draws (labels of another image, same year x style)

| source | images | own min-D beats all 20 nulls (chance 0.048) | own min-D percentile within its nulls (chance 0.5) | framed / unframed | n_det terciles low / mid / high | years |
|---|---|---|---|---|---|---|
| pre-only s46 | 8858 | **0.082** | 0.40 | 0.087 / 0.076 | 0.122 / 0.056 / 0.066 | 0.075-0.098 |
| stride-2 s49 | 8873 | 0.069 | 0.40 | 0.073 / 0.066 | 0.080 / 0.065 / 0.062 | 0.062-0.075 |

Own labels do not pick out a window better than another image's labels from the same year and style: the
beat-all rate is within 2-3 points of chance everywhere, the percentile is barely below 0.5, and the only bump
(low-detection images, 0.12) is where few windows are valid. The detector-derived (CD, HEX) surface is too smooth and
too biased (CD needs k = 1.7-3.9, i.e. 40-75 % missed cells) for a two-number label to localise anything.

## Verdict

The image-level labels cannot act as a verifier for the technician ROI or for trustworthy cells from our detections:
IoU is at the "any central window" level, the expert box ranks like a random window, and there is no separation from
the within-stratum null on the 8954. Route A (label-verified windows as pseudo cell + ROI labels for an EM loop) is
closed cleanly; it would need a point source with far higher recall before the labels could discriminate, which is
the same detection-accuracy wall E4 and E8 found.

Files: `e9_label_verifier/{protocol.json, result.json, run.log}`, `source/green_day3_e9_label_verifier_20261103.py`.
