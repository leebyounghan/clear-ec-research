"""Screen every stored member that covers folds 0 and 4 as an added CD member on top of hybrid B (CD only).
Cross-fit on the two screen folds: blend weight and stretch for fold 0 are fit on fold 4 and vice versa; the reference
(B CD) gets the same cross-fit stretch so the comparison is paired."""
import glob, json, numpy as np
from cd_common import *
pos = {int(i): k for k, i in enumerate(idx)}
Bc = E.load_base()['B'][:, 0]
def load(path, key=None):
    z = np.load(path); k = key or ('pred_tta4' if 'pred_tta4' in z.files else 'pred'); p = np.full(len(yc), np.nan)
    p[[pos[int(i)] for i in z['indices']]] = z[k][:, 0]; return p
L = 'lane_gpu{}_20261106/'
M = {}
def add(name, f0, f4):
    try:
        a = load(X / f0); b = load(X / f4); p = np.where(fold == 0, a, np.where(fold == 4, b, np.nan))
        if np.isfinite(p[SCREEN]).all(): M[name] = p
        else: print('incomplete', name, flush=True)
    except Exception as e: print('fail', name, e, flush=True)
R = 'runs/fold{}/{}/predictions_epoch_20.npz'
add('dinov3_b16_s46', 'lane_gpu3_20261106/runs/dinov3_b16_f0_s46/' + R.format(0, 'dinov3_b16'), 'lane_gpu3_20261106/runs/dinov3_b16_f4_s46/' + R.format(4, 'dinov3_b16'))
add('dinov3_b16_s47', 'lane_gpu2_20261106/runs/dinov3_b16_f0_s47/' + R.format(0, 'dinov3_b16'), 'lane_gpu2_20261106/runs/dinov3_b16_f4_s47/' + R.format(4, 'dinov3_b16'))
add('convnext_b32', 'lane_gpu3_20261106/runs/convnext_b32_f0_s46/' + R.format(0, 'convnext_b32'), 'lane_gpu3_20261106/runs/convnext_b32_f4_s46/' + R.format(4, 'convnext_b32'))
add('dinov2_b14', 'lane_gpu3_20261106/runs/dinov2_b14_f0_s46/' + R.format(0, 'dinov2_b14'), 'lane_gpu3_20261106/runs/dinov2_b14_f4_s46/' + R.format(4, 'dinov2_b14'))
add('measured_v1', 'lane_gpu0_20261106/arms/measured_v1/f0_s46/' + R.format(0, 'cpdino_vitb'), 'lane_gpu0_20261106/arms/measured_v1/f4_s46/' + R.format(4, 'cpdino_vitb'))
add('lf50', 'lane_gpu0_20261106/arms/lf50/f0_s46/' + R.format(0, 'cpdino_vitb'), 'lane_gpu0_20261106/arms/lf50/f4_s46/' + R.format(4, 'cpdino_vitb'))
add('cropopt_both', 'lane_gpu1_20261106/e2e/cropopt_both_f0/f0_s46/' + R.format(0, 'cpdino_vitb'), 'lane_gpu1_20261106/e2e/cropopt_both_f4/f4_s46/' + R.format(4, 'cpdino_vitb'))
add('s47_mean_pool', 'phase2_pooling_replication_20261026/f0_s47_mean/' + R.format(0, 'cpdino_vitb'), 'phase2_pooling_replication_20261026/f4_s47_mean/' + R.format(4, 'cpdino_vitb'))
add('s46_e20_single', 'phase2_pooling_replication_20261026/f0_s46_std/' + R.format(0, 'cpdino_vitb'), 'phase2_pooling_20261020/arms/std/' + R.format(4, 'cpdino_vitb'))
for k in list(M): print(k, 'n', int(np.isfinite(M[k]).sum()), flush=True)
def xstretch(p):
    out = np.full_like(p, np.nan)
    for f, o in ((0, 4), (4, 0)): out[fold == f] = apply1(p[fold == f], fit_stretch1(p, (fold == o) & good))
    return out
REF2 = xstretch(Bc)
print('ref cross-fit CD screen', round(cd(REF2, SCREEN), 4), 'vs 4-fold-fit', round(cd(REF, SCREEN), 4), flush=True)
lb = np.log(Bc); ly = np.log(np.where(yc > 0, yc, np.nan))
def xblend(Pm, geo=True):
    out = np.full_like(Bc, np.nan); ws = {}
    W = np.round(np.arange(0, 0.81, 0.05), 2)
    for f, o in ((0, 4), (4, 0)):
        best = None
        for w in W:
            q = np.exp((1 - w) * lb + w * np.log(Pm)) if geo else (1 - w) * Bc + w * Pm
            s = apply1(q, fit_stretch1(q, (fold == o) & good)); m = cd(s, fold == o)
            if best is None or m < best[0]: best = (m, w)
        w = best[1]; ws[f] = w
        q = np.exp((1 - w) * lb + w * np.log(Pm)) if geo else (1 - w) * Bc + w * Pm
        out[fold == f] = apply1(q[fold == f], fit_stretch1(q, (fold == o) & good))
    return out, ws
res = {}
for k, p in M.items():
    ok = SCREEN & (yc > 0)
    alone = xstretch(p); er = np.corrcoef(np.log(p[ok]) - ly[ok], lb[ok] - ly[ok])[0, 1]
    q, ws = xblend(p); r = rep(f'add {k}', q, REF2, note=f'alone {cd(alone, SCREEN):.3f} errcorr_B {er:.3f} w {ws}'); res[k] = r
groups = {'all': list(M), 'other_family': [k for k in M if k.split('_')[0] in ('dinov3', 'convnext', 'dinov2')], 'cpdino_variants': [k for k in M if k.split('_')[0] not in ('dinov3', 'convnext', 'dinov2', 'lf50')]}
for g, ks in groups.items():
    if not ks: continue
    P = np.exp(np.mean([np.log(M[k]) for k in ks], 0)); ok = SCREEN & (yc > 0); er = np.corrcoef(np.log(P[ok]) - ly[ok], lb[ok] - ly[ok])[0, 1]
    q, ws = xblend(P); res['group_' + g] = rep(f'add group {g} ({len(ks)})', q, REF2, note=f'alone {cd(xstretch(P), SCREEN):.3f} errcorr_B {er:.3f} w {ws}')
json.dump(res, open(LANE / 'reads/cd_pool_screen.json', 'w'), indent=1)
