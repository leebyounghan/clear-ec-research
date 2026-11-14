"""Read the simulation-pretrained graph network on real frames (2026-11-05). Gates in sweep/track2_protocol.json. Per detector and ensemble: r with log labels, with the residual of B + hex6 + stretch and partial after hex6; on the 25 expert frames against device values; LOFO residual fit on the predictions (HGB, L1), alone on B + hex6 + stretch and added on top of the label-trained graph ensemble; the same on the pooled embedding through ridge. usage: sim_graph_eval_20261105.py NAME"""
import json, re, sys, warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'; NAME = sys.argv[1]; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float); H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
def pipe(base):
    P = np.full_like(base, np.nan)
    for f in range(5): P[fold == f] = N.apply_stretch(base[fold == f], N.fit_stretch(base, y, (fold != f) & good))
    return P
REF = pipe(H); ly = np.log(np.where(y > 0, y, np.nan)); r = ly - np.log(REF)
R = [np.load(O / 'graph_sweep' / f'{q}.npz') for q in ('d64_l5_noctx_s10', 'd64_l5_pre47_s3', 'd64_l5_pre48_s3')]; GR = {f: np.mean([z[f'f{f}'] for z in R], 0).astype(np.float64) for f in range(5)}; G3 = H.copy()
for f in range(5):
    t = fold == f; G3[t, 1] = H[t, 1] * np.exp(GR[f][t, 1]); G3[t, 2] = H[t, 2] * np.exp(GR[f][t, 2])
G3_all = {f: H * np.exp(GR[f] * np.array([0, 1, 1])) for f in range(5)}; REF3 = np.full_like(H, np.nan)
for f in range(5): REF3[fold == f] = N.apply_stretch(G3_all[f][fold == f], N.fit_stretch(G3_all[f], y, (fold != f) & good))
def cor(a, c):
    s = np.isfinite(a) & np.isfinite(c); return round(float(np.corrcoef(a[s], c[s])[0, 1]), 3)
def res(a, x):
    s = np.isfinite(a) & np.isfinite(x); o = np.full(len(a), np.nan); A = np.c_[np.ones(s.sum()), x[s]]; o[s] = a[s] - A @ np.linalg.lstsq(A, a[s], rcond=None)[0]; return o
files = {'pre46': 'cell_graphs.npz', 'pre47': 'cell_graphs_pre47.npz', 'pre48': 'cell_graphs_pre48.npz'}; S = {}; Z = {}
for k, f in files.items():
    z = np.load(O / 'sim_pred' / f'{NAME}__{f}'); S[k] = z['pred'][idx].astype(np.float64); Z[k] = z['emb'][idx].astype(np.float64)
S['mean3'] = np.nanmean([S[k] for k in files], 0); Z['mean3'] = np.nanmean([Z[k] for k in files], 0); rep = {}
print('reference: ten-detector hex6 r log HEX', cor(f6, ly[:, 2]), 'r log CV', cor(f6, ly[:, 1]), '; window Voronoi HEX 0.47-0.51, CV 0.46, CD 0.57; label-trained graph residual r HEX 0.18, CV 0.09', flush=True)
for k, v in S.items():
    o = {K[j]: dict(r_label=cor(v[:, m], ly[:, j]), r_final_resid=cor(v[:, m], r[:, j]), partial_after_hex6=cor(res(v[:, m], f6), res(r[:, j], f6))) for j, m in ((2, 0), (1, 1), (0, 2))}; o['HEX_hat_on_CV'] = cor(v[:, 0], ly[:, 1]); rep[k] = o; print(k, json.dumps(o), flush=True)
ann = json.load(open(C.X.parent / 'datasets/clear_ec_official/green_eval/annotations.json')); pos = {n: i for i, n in enumerate(ids)}; rw = np.array([pos[n] for n in ann]); gt = np.array([[ann[n]['gt']['HEX'], ann[n]['gt']['CV'], ann[n]['gt']['CD']] for n in ann]); v = S['mean3'][rw]; print('25 expert frames, r with device: HEX', cor(v[:, 0], np.log(gt[:, 0])), 'CV', cor(v[:, 1], np.log(gt[:, 1])), 'CD', cor(v[:, 2], np.log(gt[:, 2])), '| expert markers 0.84, 0.88, 0.75; B', cor(np.log(B[rw, 2]), np.log(gt[:, 0])), cor(np.log(B[rw, 1]), np.log(gt[:, 1])), cor(np.log(B[rw, 0]), np.log(gt[:, 2])), flush=True)
def fill(X):
    X = X.copy(); med = np.nanmedian(X, 0); bad = ~np.isfinite(X); X[bad] = np.take(med, np.nonzero(bad)[1]); return X
def fit(X, base_of, ms, kind='hgb'):
    P = np.full_like(H, np.nan)
    for f in range(5):
        base = base_of(f); tr = (fold != f) & good; C_ = base.copy()
        for j in ms:
            q = tr & (y[:, j] > 0); t = np.log(y[q, j]) - np.log(base[q, j])
            if kind == 'hgb':
                m = HGB(loss='absolute_error', max_depth=3, max_iter=200, learning_rate=0.03, min_samples_leaf=80, l2_regularization=5.0, random_state=0).fit(X[q], t); C_[:, j] = base[:, j] * np.exp(m.predict(X))
            else:
                mu = X[q].mean(0); sd = X[q].std(0) + 1e-6; A = (X[q] - mu) / sd; w = np.linalg.solve(A.T @ A + 3000 * np.eye(A.shape[1]), A.T @ (t - t.mean())); C_[:, j] = base[:, j] * np.exp(((X - mu) / sd) @ w + t.mean())
        P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr))
    return P
def report(nm, p, ref):
    pb = E.paired_bootstrap(E.ape(p, y), E.ape(ref, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(ref, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(ref[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; o = dict(score=round(s['mean'], 3), per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v_, 4) for v_ in pb['ci95']], folds=per); print(nm, json.dumps(o), flush=True); return o
Hb = lambda f: H; Gb = lambda f: G3_all[f]; st = {}
for k in ('pre46', 'mean3'):
    X = fill(S[k]); st[k + ' alone CV+HEX'] = report(k + ' predictions alone, CV+HEX', fit(X, Hb, (1, 2)), REF); st[k + ' on graph'] = report(k + ' predictions on top of the graph ensemble, CV+HEX', fit(X, Gb, (1, 2)), REF3)
Xz = fill(Z['mean3']); st['emb alone'] = report('embedding ridge alone, CV+HEX', fit(Xz, Hb, (1, 2), 'ridge'), REF); st['emb on graph'] = report('embedding ridge on top of the graph ensemble, CV+HEX', fit(Xz, Gb, (1, 2), 'ridge'), REF3); print('graph ensemble reference', round(E.official(E.ape(REF3, y))['mean'], 3), flush=True)
json.dump(dict(corr=rep, stack=st), open(O / 'sim_pred' / f'{NAME}_eval.json', 'w'), indent=1); print('done', flush=True)
