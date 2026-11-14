"""CD lane screen 2: family members on top of the CD-reweighted hybrid (A2/P1 geometric blend), folds 0 and 4, cross-fit.
Components in log space: P1, A2 (recipe A two seeds, tta4), F (geometric mean of family members). Weights on a grid, fit on the other screen fold;
stretch fit on the other screen fold. Epoch-20 single view vs mean of epochs 16-20 for the family members."""
import json, numpy as np
from cd_common import *
pos = {int(i): k for k, i in enumerate(idx)}
b = E.load_base(); seeds, single, hist = C.load_members(b)
sc = np.array([json.load(open(X / 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars'][str(f)][0] for f in fold])
A2 = 0.5 * (seeds[46][:, 0] + seeds[47][:, 0]) * sc; P1 = b['P1'][:, 0]; Bc = b['B'][:, 0]
def load(path):
    z = np.load(path); p = np.full(len(yc), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; return p
def fam(d0, d4, arch, epochs):
    out = np.full(len(yc), np.nan)
    for f, d in ((0, d0), (4, d4)):
        P = np.exp(np.mean([np.log(load(X / d / f'runs/fold{f}/{arch}/predictions_epoch_{e:02d}.npz')) for e in epochs], 0)); out[fold == f] = P[fold == f]
    return out
FAM = {'dinov3_s46': ('lane_gpu3_20261106/runs/dinov3_b16_f0_s46', 'lane_gpu3_20261106/runs/dinov3_b16_f4_s46', 'dinov3_b16'),
       'dinov3_s47': ('lane_gpu2_20261106/runs/dinov3_b16_f0_s47', 'lane_gpu2_20261106/runs/dinov3_b16_f4_s47', 'dinov3_b16'),
       'convnext': ('lane_gpu3_20261106/runs/convnext_b32_f0_s46', 'lane_gpu3_20261106/runs/convnext_b32_f4_s46', 'convnext_b32'),
       'dinov2': ('lane_gpu3_20261106/runs/dinov2_b14_f0_s46', 'lane_gpu3_20261106/runs/dinov2_b14_f4_s46', 'dinov2_b14')}
M20 = {k: fam(*v, [20]) for k, v in FAM.items()}; M5 = {k: fam(*v, range(16, 21)) for k, v in FAM.items()}
ok = SCREEN & (yc > 0); ly = np.log(np.where(yc > 0, yc, np.nan))
for k in FAM: print(k, 'e20 alone raw', round(cd(M20[k], SCREEN), 3), 'e16-20', round(cd(M5[k], SCREEN), 3), flush=True)
def xs(p):
    out = np.full_like(p, np.nan)
    for f, o in ((0, 4), (4, 0)): out[fold == f] = apply1(p[fold == f], fit_stretch1(p, (fold == o) & good))
    return out
def xblend(comps, grids):
    """comps: list of CD vectors; grids: list of weight tuples (sum to 1). Choose on the other fold."""
    L = [np.log(c) for c in comps]; out = np.full_like(Bc, np.nan); ws = {}
    cache = {}
    for wt in grids:
        q = np.exp(sum(w * l for w, l in zip(wt, L))); cache[wt] = q
    for f, o in ((0, 4), (4, 0)):
        best = None
        for wt, q in cache.items():
            m = cd(apply1(q, fit_stretch1(q, (fold == o) & good)), fold == o)
            if best is None or m < best[0]: best = (m, wt)
        wt = best[1]; ws[f] = wt; q = cache[wt]; out[fold == f] = apply1(q[fold == f], fit_stretch1(q, (fold == o) & good))
    return out, ws
R1 = xs(Bc)
g2 = [(round(1 - a, 2), round(a, 2)) for a in np.arange(0.3, 1.0001, 0.05)]
R2, w2 = xblend([P1, A2], g2); rep('R2 = P1/A2 geo reweight', R2, R1, note=str(w2))
res = {}
for tag, MM in (('e20', M20), ('e16_20', M5)):
    F = np.exp(np.mean([np.log(MM[k]) for k in FAM], 0))
    g3 = [(round(1 - a - c, 2), round(a, 2), round(c, 2)) for a in np.arange(0.2, 0.901, 0.05) for c in np.arange(0, 0.701, 0.05) if a + c <= 1.0001]
    p3, w3 = xblend([P1, A2, F], g3)
    res[tag + '_vs_R1'] = rep(f'P1/A2/F4 {tag} vs B', p3, R1, note=str(w3)); res[tag + '_vs_R2'] = rep(f'P1/A2/F4 {tag} vs R2', p3, R2, note=str(w3))
    for k in FAM:
        p, w = xblend([P1, A2, MM[k]], g3); rep(f'P1/A2/{k} {tag} vs R2', p, R2, note=str(w))
    for drop in FAM:
        F3 = np.exp(np.mean([np.log(MM[k]) for k in FAM if k != drop], 0)); p, w = xblend([P1, A2, F3], g3); rep(f'P1/A2/F3 without {drop} {tag} vs R2', p, R2, note=str(w))
json.dump(res, open(LANE / 'reads/cd_pool_screen2.json', 'w'), indent=1)
