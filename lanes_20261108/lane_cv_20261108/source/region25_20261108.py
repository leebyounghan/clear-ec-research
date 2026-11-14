"""CV lane: on the 25 expert frames, where does point-CV signal go? area CV / log-area SD r with device CV for peak sets (t0.35 plain,
spacing-prior NMS a0.7 t0.2) read in the marker hull, the box, top-1 / top-3 legible windows (380x300), box-size windows, whole frame."""
import sys
import numpy as np
from scipy.spatial import Delaunay, cKDTree
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cv_20261108/source')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import peak_select25_20261108 as P
import legible_window_features_20261105 as L
ann = P.ann; names = P.names
rows = []
for n, mk in zip(names, P.MK):
    r = P.pos[n]; pk = P.PK[int(P.idx[r])]; s = P.spacing_from_cd(P.B[r, 0]); hi = pk[pk[:, 2] >= 0.5, :2].astype(float)
    sets = {'t035': pk[pk[:, 2] >= 0.35, :2].astype(float), 'sp07': P.select(pk, s, t_hi=0.2, alpha=0.7)}
    box = ann[n].get('roi_box_mha'); o = {}
    for k, q in sets.items():
        hull = Delaunay(mk); o[k + ' hull'] = P.stats(q[hull.find_simplex(q) >= 0])
        if box is not None:
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            m = (q[:, 0] >= x0) & (q[:, 0] < x1) & (q[:, 1] >= y0) & (q[:, 1] < y1); o[k + ' box'] = P.stats(q[m])
        for nm, (W, H, K) in {'win1': (380, 300, 1), 'win3': (380, 300, 3), 'boxwin1': (538, 408, 1), 'boxwin2': (538, 408, 2)}.items():
            acc = []
            for x0, y0 in L.windows(hi, W, H, k=K):
                m = (q[:, 0] >= x0) & (q[:, 0] < x0 + W) & (q[:, 1] >= y0) & (q[:, 1] < y0 + H); acc.append(P.stats(q[m]))
            o[k + ' ' + nm] = {e: float(np.nanmean([a.get(e, np.nan) for a in acc])) for e in ('area_cv', 'log_area_sd', 'edge_cv', 'nn_cv', 'area_mad')}
        o[k + ' frame'] = P.stats(q)
    o['markers hull'] = P.stats(mk); rows.append(o)
print('box example', ann[names[0]].get('roi_box_mha'), 'hull area px2 median', np.median([Delaunay(m).volume if hasattr(Delaunay(m), 'volume') else 0 for m in P.MK]))
for k in rows[0]:
    print(k.ljust(16), P.read([o.get(k, {}) for o in rows]), flush=True)
