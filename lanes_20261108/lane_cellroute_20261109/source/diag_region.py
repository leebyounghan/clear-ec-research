"""Cell-route lane (2026-11-09): which cells to read for CV. Union point set P2 (and seeds3 t0.2); Konan k=2 cells read under selection rules,
on the 25 marker frames. Reference: marker hull (the expert's own patch). Rules (all automatic, inside the predicted 538x408 box):
box = all interior cells; trim = drop cells with area outside [0.4, 2.5] x median; conf = cell and all neighbours vote share >= 0.5;
patchN = the N interior cells nearest the box centre of mass of confident peaks; patch_conf = patch150 among conf cells."""
import sys
import numpy as np
from scipy.spatial import Delaunay, cKDTree
sys.path[:0] = ['/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source', '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source']
import gate25 as G
import cr_lib as L
import legible_window_features_20261105 as LW
SETS = {'P2': dict(src='vote', members=['s46_tta4', 's47_tta4', 's48_tta4', 'seg'], minv=1), 'seeds3 t0.2': dict(src='seeds3_tta4', t=0.2)}
RULES = ['hull', 'box', 'box trim', 'box conf', 'box conf trim', 'patch100', 'patch150', 'patch150 trim', 'patch150 conf', 'hullpatch150']
for sn, v in SETS.items():
    R = {r: [] for r in RULES}
    for i in range(25):
        pk = G.PEAKS['seeds3_tta4'][G.DS[i]]; hi = pk[pk[:, 2] >= 0.5, :2]; x0, y0 = LW.windows(hi, 538, 408, k=1)[0]
        p, w, _, _ = G.points(i, v); ix, res, nb = L.konan_cells_idx(p)
        a2, s2 = res[2]; pts = p[ix]; inbox = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + 538) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + 408)
        conf = np.array([w[j] >= 0.5 and (w[nb[j]] >= 0.5).all() for j in ix]) if v['src'] == 'vote' else np.array([w[j] >= 0.35 and (w[nb[j]] >= 0.35).all() for j in ix])
        hull = Delaunay(G.MK[i]).find_simplex(pts) >= 0
        hb = hi[(hi[:, 0] >= x0) & (hi[:, 0] < x0 + 538) & (hi[:, 1] >= y0) & (hi[:, 1] < y0 + 408)]; c0 = hb.mean(0) if len(hb) else np.array([x0 + 269, y0 + 204])
        d = np.linalg.norm(pts - c0, axis=1); dh = np.linalg.norm(pts - G.MK[i].mean(0), axis=1)
        def trim(m):
            if m.sum() < 10: return m
            med = np.median(a2[m]); return m & (a2 > 0.4 * med) & (a2 < 2.5 * med)
        def near(m, N, dd):
            o = np.where(m)[0]; o = o[np.argsort(dd[o])][:N]; mm = np.zeros(len(m), bool); mm[o] = True; return mm
        masks = {'hull': hull, 'box': inbox, 'box trim': trim(inbox), 'box conf': inbox & conf, 'box conf trim': trim(inbox & conf), 'patch100': near(inbox, 100, d),
                 'patch150': near(inbox, 150, d), 'patch150 trim': trim(near(inbox, 150, d)), 'patch150 conf': near(inbox & conf, 150, d), 'hullpatch150': near(np.ones(len(pts), bool), 150, dh)}
        for r, m in masks.items():
            if m.sum() < 10: R[r].append([np.nan, np.nan]); continue
            R[r].append([a2[m].std() / a2[m].mean(), np.mean(s2[m] == 6)])
    for r in RULES:
        A = np.array(R[r], float); print(f'{sn:12s} {r:15s}', 'CV k2', G.read(A[:, 0], G.gt_cv), '| HEX k2', G.read(A[:, 1], G.gt_hex, log=False), flush=True)
