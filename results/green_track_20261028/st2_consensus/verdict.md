# st2_consensus — verdict (2026-10-29)

Not extended. Consensus pseudo-points (>= 3 of 4 fold-clean detectors within 8 px; median 169 per training
image, 19 images without any) plus the 20 strong images train a stride-2 student that reaches val-5 F1 0.679
vs 0.666 for the seed-46 head (+0.012; rule required >= 0.70). Its hex6 correlates slightly better with GT HEX
on the fold-4 validation rows (r 0.489 vs 0.454) but also more with the base's errors (0.66 vs 0.58). The
consensus teacher re-teaches what the seeds already agree on; self-training on this track is closed.
