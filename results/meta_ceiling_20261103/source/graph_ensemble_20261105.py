"""Averages cell-graph predictions across runs in the log-residual space and reads them in the deployed pipeline (2026-11-05). Also with the HGB window fit on HEX, and on the re-selected members. usage: graph_ensemble_20261105.py NAME:run1,run2,... ... Runs are files in sweep/graph_sweep. Reference: B + hex6 + stretch."""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold = b['y'], b['B'], b['fold']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float)
def hex6(base):
    H = base.copy(); H[have, 2] = np.exp((1 - w_) * np.log(base[have, 2]) + w_ * (a_ + b_ * f6[have])); return H
H = hex6(B); c = np.load(O / 'combo_preds.npz'); key = [k for k in c.files if k.startswith('bagged_per_metric_from_B')][0]; HM = hex6(c[key].astype(np.float64))
def pipe(base, pr, ms):
    P = np.full_like(base, np.nan)
    for f in range(5):
        tr = (fold != f) & good; C_ = base.copy()
        for j in ms: C_[:, j] = base[:, j] * np.exp(pr[f][:, j])
        P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
    return P
REF = pipe(H, {f: np.zeros_like(H) for f in range(5)}, ())
def report(nm, p):
    pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(REF, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(REF[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; o = dict(score=round(s['mean'], 3), per={k: round(s[k], 3) for k in K}, per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm, json.dumps(o), flush=True); return o
rep = {}
for spec in sys.argv[1:]:
    nm, runs = spec.split(':'); R = [np.load(O / 'graph_sweep' / f'{r}.npz') for r in runs.split(',')]; pr = {f: np.mean([z[f'f{f}'] for z in R], 0).astype(np.float64) for f in range(5)}
    rep[nm + ' CV+HEX'] = report(nm + ' CV+HEX', pipe(H, pr, (1, 2))); rep[nm + ' all'] = report(nm + ' all', pipe(H, pr, (0, 1, 2))); rep[nm + ' CV+HEX members'] = report(nm + ' CV+HEX on re-selected members', pipe(HM, pr, (1, 2)))
json.dump(rep, open(O / 'graph_ensemble.json', 'w'), indent=1); print('done', flush=True)
