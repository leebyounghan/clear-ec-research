# p10_official_hex — verdict (2026-10-30)

The organiser's HEX (calculate_hexagonality: mean IoU of each cell mask with a hexagon fitted from the cell's
ellipse axes x0.8 and orientation) cannot be reproduced from points. Applied to label masks of bounded Voronoi
cells of the 25 expert point sets it is nearly constant (0.624-0.636 while GT HEX spans 0.31-0.67), correlates
-0.81 with GT HEX and -0.85 with hex6, and gives 15.8% LOO-scaled MAPE against 7.8% for hex6 (r +0.845). Voronoi
polygons are convex and hexagon-like by construction, so the official shape score loses the irregularity that
real cell outlines carry; the six-neighbour fraction is the right point-derived proxy and stays as the feature.
Recorded with the official metric facts (pix_to_um 1000/1296, frame 0.75 mm^2) in the ledger.
