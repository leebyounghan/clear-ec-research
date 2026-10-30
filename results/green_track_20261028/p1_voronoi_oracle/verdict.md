# p1_voronoi_oracle — verdict (2026-10-28)

With the 25 expert marker sets, bounded Voronoi cells recover the GT far better than the regressor for CV and
HEX: LOO-scaled MAPE CV 8.8% (r 0.88), HEX via six-neighbour fraction 7.8% (r 0.84), CD 7.2%. The raw Voronoi
area CV is biased low (ratio 0.49) but linear. The prize exists only with near-perfect points: restricting to
complete neighbourhoods already costs CV (11.5%, r 0.55) because the irregular cells are the CV signal, and
15% random misses plus 15% false positives collapse the CV correlation to 0.08 and HEX to 0.38. Real
detectors are at F1 0.65 on the 5 validation images, so this route needs roughly F1 >= 0.9.
