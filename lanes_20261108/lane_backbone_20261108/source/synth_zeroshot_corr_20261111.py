"""Correlations of the zero-shot synthetic-pretrained columns with our labels (backbone lane, 2026-11-11; measurement only), overall
and by source, next to label-free comparators: zero-shot CPSAM masks + Konan rule (lane_cellroute feats/seg_konan_feats.npz, window
raw: cd_k0, cv_k2, hex_k0) and the detector legible window (meta_ceiling legible_window.npz top_box cd / cv_area / hex6).
Pearson r on logs and Spearman rho, rows with three positive labels and finite features."""
import json, numpy as np
from scipy.stats import spearmanr
X = '/workspace/clear-ec-research/experiments/'
with np.load('/workspace/clear-ec-research/datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
    Y = z['Y'].astype(float)
zs = np.load(X + 'lane_backbone_20261108/data/synth_zeroshot_f0pre_20261111.npz', allow_pickle=True); assert (zs['idx'] == np.arange(len(Y))).all()
Z = zs['pred'].astype(float)
sk = np.load(X + 'lane_cellroute_20261109/feats/seg_konan_feats.npz', allow_pickle=True); skn = list(sk['names'])
SK = np.stack([sk['F'][:, skn.index(c)] for c in ('seg_win_raw_cd_k0', 'seg_win_raw_cv_k2', 'seg_win_raw_hex_k0')], 1).astype(float)
lw = np.load(X + 'meta_ceiling_20261103/sweep/legible_window.npz', allow_pickle=True); lwn = list(lw['names'])
LW = np.stack([lw['F'][:, lwn.index(c)] for c in ('top_box_cd', 'top_box_cv_area', 'top_box_hex6')], 1).astype(float)
sp = np.load(X + 'lane_backbone_20261108/data/source_split_20261110.npz', allow_pickle=True)
groups = {'all': np.arange(len(Y)), 'P1': sp['P1'], 'P2': sp['P2']}
out = {}
for gname, rows in groups.items():
    out[gname] = {}
    for fname, M in (('synth_zeroshot', Z), ('cpsam_konan', SK), ('detector_window', LW)):
        out[gname][fname] = {}
        for k, met in enumerate(('CD', 'CV', 'HEX')):
            y, f = Y[rows, k], M[rows, k]; ok = np.isfinite(y) & (y > 0) & np.isfinite(f) & (f > 0)
            out[gname][fname][met] = dict(n=int(ok.sum()), r_log=round(float(np.corrcoef(np.log(y[ok]), np.log(f[ok]))[0, 1]), 3), rho=round(float(spearmanr(y[ok], f[ok]).correlation), 3))
print(json.dumps(out))
open(X + 'lane_backbone_20261108/reads/synth_zeroshot_corr_20261111.json', 'w').write(json.dumps(out, indent=1))
