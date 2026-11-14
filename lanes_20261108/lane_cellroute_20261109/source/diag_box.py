"""Cell-route lane (2026-11-09): can the technician's analysis box (fixed 538x408, movable) be found from the image's own peaks?
Candidate = the 538x408 window with the most confident peaks (>= 0.5), stride 16 (LW.windows k=1). Read: IoU with roi_box_mha on the 25 frames,
and Konan k=2 HEX/CV of the P2 union in the predicted box vs in the true box vs the 3 legible 380x300 windows."""
import sys
import numpy as np
from scipy.spatial import cKDTree
sys.path[:0] = ['/workspace/clear-ec-research/experiments/lane_cellroute_20261109/source', '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source']
import gate25 as G
import cr_lib as L
import legible_window_features_20261105 as LW
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I)
v = dict(src='vote', members=['s46_tta4', 's47_tta4', 's48_tta4', 'seg'], minv=1); R = []
for i in range(25):
    tb = G.P.ann[G.names[i]]['roi_box_mha']; pk = G.PEAKS['seeds3_tta4'][G.DS[i]]; hi = pk[pk[:, 2] >= 0.5, :2]
    x0, y0 = LW.windows(hi, 538, 408, k=1)[0]; pb = [x0, y0, x0 + 538, y0 + 408]; wins = LW.windows(hi, 380, 300, k=3)
    p, w, _, _ = G.points(i, v); out = [iou(tb, pb)]
    for reg in ('true', 'pred', 'wins'):
        if reg == 'wins':
            m = np.zeros(len(p), bool)
            for a, b in wins: m |= (p[:, 0] >= a) & (p[:, 0] < a + 380) & (p[:, 1] >= b) & (p[:, 1] < b + 300)
        else:
            bx = tb if reg == 'true' else pb; m = (p[:, 0] >= bx[0]) & (p[:, 0] < bx[2]) & (p[:, 1] >= bx[1]) & (p[:, 1] < bx[3])
        res = L.konan_cells(p, m); cv2, hx2, n = L.summarize(res, 2); out += [cv2, hx2]
    R.append(out)
R = np.array(R, float)
print('IoU predicted vs technician box: median %.2f, share >= 0.5: %.2f' % (np.median(R[:, 0]), np.mean(R[:, 0] >= 0.5)))
for j, reg in ((1, 'technician box'), (3, 'predicted box'), (5, '3 legible windows')):
    print(reg.ljust(18), 'CV k2', G.read(R[:, j], G.gt_cv), '| HEX k2', G.read(R[:, j + 1], G.gt_hex, log=False))
