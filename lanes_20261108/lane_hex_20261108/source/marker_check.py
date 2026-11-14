"""Point quality of single detectors vs consensus points on the 25 expert-marker frames (HEX lane, 2026-11-08).
Inside the marker ROI box: miss, false, jitter (/ marker spacing) by Hungarian matching within 0.5 spacing; hex6 inside the
marker hull vs marker hex6. Frames a detector saw in training are left out for that detector."""
import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, Delaunay
sys.argv = [sys.argv[0], 'x', 'all16']
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import feat_fuse as FF
ann = json.load(open('/workspace/clear-ec-research/datasets/clear_ec_official/green_eval/annotations.json'))
G = FF.G; F0 = '/workspace/clear-ec-research/experiments/green_track_20261028/external_srep/preonly'
idmap = {}; seen = {}
for d in FF.SETS['all16'] + ['FUSED_all16', 'FUSED_pre3', 'FUSED_sg9']:
    pass


def loadset(path):
    out = {}; sn = {}
    for f in range(5):
        z = np.load(f'{path}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True); pk, of, ids = z['peaks'], z['offsets'], z['ids'] if 'ids' in z.files else None
        if ids is None:
            ids = np.load(f'{F0}/fold{f}/fold{f}_sg2_peaks.npz', allow_pickle=True)['ids']
        pls = z['point_labels_seen'] if 'point_labels_seen' in z.files else np.zeros(len(ids), bool)
        for k, i in enumerate(ids):
            i = str(i).strip()
            if i in ann: out[i] = pk[of[k]:of[k + 1]]; sn[i] = bool(pls[k])
    return out, sn


def quality(pk, thr, name):
    rows = []
    for i, a in ann.items():
        if i not in pk[0] or pk[1].get(i): continue
        M = np.array(a['markers_mha'], float)[:, :2]; x0, y0, x1, y1 = np.array(a['roi_box_mha'], float)[:4] if len(a['roi_box_mha']) == 4 else (None,) * 4
        sp = float(np.median(cKDTree(M).query(M, k=2)[0][:, 1]))
        q = pk[0][i]; q = q[q[:, 2] >= thr][:, :2].astype(float)
        hull = Delaunay(M); inb = hull.find_simplex(q) >= 0; Q = q[inb]
        D = np.linalg.norm(M[:, None] - Q[None], axis=2) if len(Q) else np.zeros((len(M), 0))
        r, c = linear_sum_assignment(np.where(D < 0.5 * sp, D, 1e6)) if len(Q) else (np.array([], int), np.array([], int))
        ok = D[r, c] < 0.5 * sp if len(Q) else np.array([], bool); tp = ok.sum()
        miss = 1 - tp / len(M); false = 1 - tp / max(len(Q), 1); jit = float(np.mean(D[r[ok], c[ok]]) / sp) if tp else np.nan
        hm = FF.hexstats(M)[0]; hq = FF.hexstats(q[Delaunay(M).find_simplex(q) >= 0])[0]
        rows.append([miss, false, jit, hm, hq, a['gt'][2] if isinstance(a['gt'], list) else np.nan])
    R = np.array(rows, float)
    if not len(R): print(name, 'no frames'); return
    ok = np.isfinite(R[:, 4])
    print(f'{name:34s} thr {thr:.2f} n {len(R):2d} miss {R[:,0].mean():.3f} false {R[:,1].mean():.3f} jitter {np.nanmean(R[:,2]):.3f}  hex6 r vs markers {np.corrcoef(R[ok,3],R[ok,4])[0,1]:.3f} MAE {np.mean(np.abs(R[ok,3]-R[ok,4])):.3f} bias {np.mean(R[ok,4]-R[ok,3]):+.3f}', flush=True)


k0 = list(ann)[0]; print('gt field', ann[k0]['gt'], 'roi', ann[k0]['roi_box_mha'])
for d in ['external_srep/preonly', 'external_srep/s47_preonly', 'sg2_specialist', 'sg2_specialist_s49', 'frozen_heads/sg2pref_s46']:
    pk = loadset(G + d)
    for thr in (0.35, 0.5): quality(pk, thr, d)
L = '/workspace/clear-ec-research/experiments/lane_hex_20261108/feats/'
import os
for d in sorted(os.listdir(L)):
    if d.startswith('fpk_'):
        pk = loadset(L + d)
        for thr in (0.2, 0.3, 0.4, 0.5, 0.6): quality(pk, thr, d)
