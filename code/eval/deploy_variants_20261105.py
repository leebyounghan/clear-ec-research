"""What can ship with the least new training (2026-11-05)? Graph on CV and HEX with each detector subset, since pre-only s46 is the only pre-only detector in the current image; and recipe A share per metric as the zero-training member change, with and without the graph. Per-metric weights come from the other folds. Reference: B + hex6 + stretch."""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold = b['y'], b['B'], b['fold']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'; A, P1 = b['A'], b['P1']
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float)
def hex6(base):
    H = base.copy(); H[have, 2] = np.exp((1 - w_) * np.log(base[have, 2]) + w_ * (a_ + b_ * f6[have])); return H
def pipe(H_, pr=None, ms=(1, 2)):
    P = np.full_like(H_, np.nan)
    for f in range(5):
        tr = (fold != f) & good; C_ = H_.copy()
        if pr is not None:
            for j in ms: C_[:, j] = H_[:, j] * np.exp(pr[f][:, j])
        P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
    return P
def load(names):
    R = [np.load(O / 'graph_sweep' / f'{r}.npz') for r in names]; return {f: np.mean([z[f'f{f}'] for z in R], 0).astype(np.float64) for f in range(5)}
REF = pipe(hex6(B))
def rep(nm, p):
    pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(REF, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(REF[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; print(nm, json.dumps(dict(score=round(s['mean'], 3), CD=round(s['CD'], 3), CV=round(s['CV'], 3), HEX=round(s['HEX'], 3), delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per)), flush=True)
s46, s47, s48 = 'd64_l5_noctx_s10', 'd64_l5_pre47_s3', 'd64_l5_pre48_s3'
for nm, v in {'graph s46 only': [s46], 'graph s46+s47': [s46, s47], 'graph s46+s48': [s46, s48], 'graph s47+s48': [s47, s48], 'graph x3': [s46, s47, s48]}.items(): rep(nm, pipe(hex6(B), load(v)))
W = np.full_like(B, np.nan); ws = []
for f in range(5):
    tr = (fold != f) & good; o = B.copy(); w0 = []
    for j in range(3):
        q = tr & (y[:, j] > 0); gr = np.linspace(0, 1, 21); e = [np.mean(np.abs(w * A[q, j] + (1 - w) * P1[q, j] - y[q, j]) / y[q, j]) for w in gr]; w = gr[int(np.argmin(e))]; o[:, j] = w * A[:, j] + (1 - w) * P1[:, j]; w0.append(round(float(w), 2))
    W[fold == f] = o[fold == f]; ws.append(w0)
print('recipe A share per metric, per fold (CD, CV, HEX)', ws, flush=True); rep('metric weights, no graph', pipe(hex6(W))); pr = load([s46, s47, s48]); rep('metric weights + graph x3', pipe(hex6(W), pr)); rep('metric weights + graph s46+s48', pipe(hex6(W), load([s46, s48])))
Wcv = B.copy(); Wcd = B.copy()
for f in range(5): Wcd[fold == f, 0] = W[fold == f, 0]
rep('weight on CD only + graph x3', pipe(hex6(Wcd), pr)); print('done', flush=True)
