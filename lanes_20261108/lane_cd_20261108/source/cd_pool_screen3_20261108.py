"""CD lane screen 3 (folds 0, 4, cross-fit): per-source blend weights and stretch; detector density estimate as a fourth component.
Components (log space): P1, A2, F = geometric mean of DINOv3 s46/s47 + ConvNeXt + DINOv2 (epochs 16-20), D = detector s48 top-3 triangle median,
calibrated to log CD per source on folds != f (label-free estimate, so this uses only training labels)."""
import json, numpy as np
from cd_common import *
import cd_pool_screen2_20261108 as P2   # reuses its arrays (runs its prints once)
P1, A2, Bc = P2.P1, P2.A2, P2.Bc
F = np.exp(np.mean([np.log(P2.M5[k]) for k in P2.FAM], 0))
z = np.load(X / 'cd_density_channel_20261106/features.npz'); names = [str(n) for n in z['names']]; Dr = z['F'][idx].astype(float)[:, names.index('s48_top3_tri_med')]
ly = np.log(np.where(yc > 0, yc, np.nan))
def calibD(f):
    out = np.full(len(yc), np.nan)
    for s in (False, True):
        q = (fold != f) & good & (yc > 0) & (SRC_NEW == s) & np.isfinite(Dr); co = np.polyfit(Dr[q], ly[q], 1); m = SRC_NEW == s; out[m] = np.exp(np.polyval(co, Dr[m]))
    return out
D = np.full(len(yc), np.nan)
for f in (0, 4): D[fold == f] = calibD(f)[fold == f]
print('D finite on screen', int(np.isfinite(D[SCREEN]).sum()), 'D alone', round(cd(np.where(np.isfinite(D), D, Bc), SCREEN), 3), flush=True)
D = np.where(np.isfinite(D), D, Bc)
def blend_eval(comps, grids, per_source):
    L = [np.log(c) for c in comps]; out = np.full_like(Bc, np.nan); ws = {}
    Q = {wt: np.exp(sum(w * l for w, l in zip(wt, L))) for wt in grids}
    groups = ((False, SRC_NEW == False), (True, SRC_NEW == True)) if per_source else ((None, np.ones(len(yc), bool)),)
    for f, o in ((0, 4), (4, 0)):
        for s, m in groups:
            best = None
            for wt, q in Q.items():
                prm = fit_stretch1(q, (fold == o) & good & m); e = cd(apply1(q, prm), (fold == o) & m)
                if best is None or e < best[0]: best = (e, wt, prm)
            _, wt, prm = best; ws[(f, s)] = wt; sel = (fold == f) & m; out[sel] = apply1(Q[wt][sel], prm)
    return out, ws
R1 = P2.R1
g2 = [(round(1 - a, 2), round(a, 2)) for a in np.arange(0.3, 1.0001, 0.05)]
g3 = [(round(1 - a - c, 2), round(a, 2), round(c, 2)) for a in np.arange(0.2, 0.901, 0.05) for c in np.arange(0, 0.701, 0.05) if a + c <= 1.0001]
g4 = [(round(1 - a - c - d, 2), round(a, 2), round(c, 2), d) for a in np.arange(0.2, 0.801, 0.05) for c in np.arange(0, 0.601, 0.05) for d in (0, 0.05, 0.1, 0.15, 0.2) if a + c + d <= 1.0001]
R2, _ = blend_eval([P1, A2], g2, False)
for ps in (False, True):
    p, w = blend_eval([P1, A2, F], g3, ps); rep(f'P1/A2/F per_source={ps} vs B', p, R1, note=str(w))
    p, w = blend_eval([P1, A2, F, D], g4, ps); rep(f'P1/A2/F/D per_source={ps} vs B', p, R1, note=str(w))
    p, w = blend_eval([P1, A2, D], [(a, b_, d) for (a, b_, c, d) in g4 if c == 0], ps); rep(f'P1/A2/D per_source={ps} vs R2', p, R2, note=str(w))
