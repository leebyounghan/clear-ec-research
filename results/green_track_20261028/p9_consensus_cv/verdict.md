# p9_consensus_cv — verdict (2026-10-29)

Negative. Consensus points of the 13 detectors (k = 5/7/9, 165-220 per image) give Voronoi area CV with
calibrated MAPE 14.3-14.5% and error correlation 0.77 with the base; LOFO weight 0 on every fold, bases A and
B. Agreement removes false positives but not the misses that merge cells, so the area distribution stays
corrupted; CV from points is closed. Consensus hex6 behaves like a 5-detector hex6 (A -0.077 pp HEX).
