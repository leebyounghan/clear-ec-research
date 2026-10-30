# p4_hexshape — verdict (2026-10-28)

Primary fails: fitted-regular-hexagon IoU (mean over bounded Voronoi cells) gives -0.008 pp HEX (3/5 folds);
the continuous shape detail is noise at this detection quality. Exploratory best: mean |sides - 6|
(-0.068 pp stride-2, -0.056 stride-8, 5/5 folds), slightly better than the six-neighbour fraction
(-0.063 / -0.054); frac_sides5 -0.034; edge-length CV -0.028. Treated as exploratory: it is carried into the
p6 four-detector ensemble as a secondary feature and only counts if it beats hex6_all there.
