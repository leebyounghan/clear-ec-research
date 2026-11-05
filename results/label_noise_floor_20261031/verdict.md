# label_noise_floor_20261031 — verdict (2026-10-31, CPU)

Question (plan of 2026-10-30): is there headroom below our error, i.e. should the remaining days go to new-idea
search or to finishing known increments?

| Metric | Hybrid B OOF MAPE | Floor lower bound (ROI half-split + rounding) | Shared-error upper bound (A vs Phase-1) |
|---|---:|---:|---:|
| CD | 6.31 | 3.9-4.7 | 6.16 |
| CV | 10.20 | not measurable (proxy fails: 15.8-17.1 > observed error) | 10.12 |
| HEX | 10.12 | 8.6 | 10.04 |

Reading:
- HEX is close to its floor: a predictor that cannot know where the technician placed the ROI cannot go below
  ~8.6 % (binomial sampling of ~115 bounded cells alone gives ~6.7 %). Headroom <= 1.5 pp, likely less.
- CD has room in principle (~1.6-2.4 pp) but every map-based CD estimator so far added nothing (p0).
- CV: Voronoi-area CV of the markers is far noisier than the device CV (and biased 0.49x), so the half-split proxy
  is invalid for CV; no bound.
- Recombining the two existing predictors is capped at ~0.1 pp mean: 95-98 % of the hybrid's error variance is
  shared by the two very different models. Post-hoc combination ideas (per-metric weights, disagreement gating)
  can at best take a fraction of that.
- The only route below the HEX sampling floor is knowing which cells were measured: the ROI box is a fixed
  538 x 408 px (17 % of the frame) but its centre moves (sd 137 px in x, 92 px in y), so ROI-aware pooling is a
  research idea, not something that can be trained and five-fold validated before the 11-09 freeze.

Decision consequence: stay on the plan (finish hex6 and SSL-seed increments for submission 2); no new exploration
track before the deadline; ROI-aware pooling goes to post-challenge research.

Side result: 0 of 6,768 donors (slide_id, two eyes) span two folds. Correction 2026-11-04: this holds only because old-format IDs (`nnnn-yy`, 4514 rows) carry `slide_id` = ID, so each counts as its own donor. Adjacent old-format IDs behave like fellow eyes (CD label correlation 0.48) and share a fold at the random rate (0.187), so the folds are donor-grouped for new-format IDs only. Patient-level independence of the folds stays unverified for old-format rows.

Caveats: the lower bound uses Voronoi metrics of the technician markers as a proxy for the device metrics and ignores
ROI-placement heterogeneity, device segmentation noise and operator variability (all would raise the floor); the
upper bound attributes all shared error to labels (shared model bias would also sit there). n = 25 images.
