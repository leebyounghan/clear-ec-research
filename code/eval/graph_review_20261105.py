"""Review of the current best pipeline before planning (2026-11-05). All on stored OOF predictions.
  A selection optimism: the graph configuration chosen in each fold from the other four folds, read on the held fold, against the configuration picked after looking at all folds.
  B transfer: the same graph correction on other bases (recipe A alone, crop members alone, each seed, the deployed crop six on folds 2-4): does the gain depend on hybrid B's exact residual?
  C what it learned: spread and calibration of the predicted correction; correlation with window statistics, whole-frame hex6 and B's predictions; where the gain sits (size of the correction, HEX tertile, source, cell count); why fold 2 gains least.
Reference for every delta: the same base + hex6 + stretch without the graph."""
import json, re, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); O = C.X / 'meta_ceiling_20261103/sweep'; N_ = len(y); pos = {int(i): k for k, i in enumerate(idx)}; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]
fmt = np.array([0 if re.fullmatch(r'\d{4}-\d{2}', i) else 1 for i in ids]); ly = np.log(np.where(y > 0, y, np.nan))
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float)
def hex6(base):
    H = base.copy(); H[have, 2] = np.exp((1 - w_) * np.log(base[have, 2]) + w_ * (a_ + b_ * f6[have])); return H
def pipe(H_, pr=None, ms=(1, 2), rows=None):
    P = np.full_like(H_, np.nan); use = np.ones(N_, bool) if rows is None else rows
    for f in np.unique(fold[use]):
        tr = (fold != f) & good & use if rows is not None else (fold != f) & good; C_ = H_.copy()
        if pr is not None:
            for j in ms: C_[:, j] = H_[:, j] * np.exp(pr[int(f)][:, j])
        te = (fold == f) & use; P[te] = N.apply_stretch(C_[te], N.fit_stretch(C_, y, tr))
    return P
H = hex6(B); REF = pipe(H)
def load(names):
    R = [np.load(O / 'graph_sweep' / f'{r}.npz') for r in names]; return {f: np.mean([z[f'f{f}'] for z in R], 0).astype(np.float64) for f in range(5)}
def rep(nm, p, ref, m=None):
    m = np.isfinite(p).all(1) & np.isfinite(ref).all(1) if m is None else m; pb = E.paired_bootstrap(E.ape(p[m], y[m]), E.ape(ref[m], y[m]), np.unique(g[m], return_inverse=True)[1]); s, s0 = E.official(E.ape(p[m], y[m])), E.official(E.ape(ref[m], y[m])); per = [round(E.official(E.ape(p[m & (fold == f)], y[m & (fold == f)]))['mean'] - E.official(E.ape(ref[m & (fold == f)], y[m & (fold == f)]))['mean'], 3) if (m & (fold == f)).sum() else None for f in range(5)]; o = dict(n=int(m.sum()), base=round(s0['mean'], 3), score=round(s['mean'], 3), CV=round(s['CV'] - s0['CV'], 3), HEX=round(s['HEX'] - s0['HEX'], 3), delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm, json.dumps(o), flush=True); return o
out = {}
print('=== A selection optimism', flush=True)
SINGLE = ['d64_l1_noctx_s3', 'd64_l3_noctx_s3', 'd64_l3_noctx_s10', 'd64_l3_noctx_e50_s3', 'd64_l3_noctx_lr3e4_s3', 'd64_l5_noctx_s3', 'd64_l5_noctx_s10', 'd64_l7_noctx_s3', 'd64_l9_noctx_s3', 'd128_l3_noctx_s3', 'd64_l5_k1_s3', 'd64_l5_k5_s3', 'd64_l5_whole_s3', 'd64_l5_pre47_s3', 'd64_l5_pre48_s3', 'd64_l5_sg2_s3', 'edge_geom_46', 'edge_ridge_46', 'edge_all_46']
CFG = {n: [n] for n in SINGLE}; PREX3 = ['d64_l5_noctx_s10', 'd64_l5_pre47_s3', 'd64_l5_pre48_s3']; CFG.update({'pre46+47': PREX3[:2], 'pre47+48': PREX3[1:], 'pre x3': PREX3})
PP = {n: pipe(H, load(v)) for n, v in CFG.items()}; err = {n: E.ape(PP[n], y) for n in CFG}
def off(e, m): return float(np.mean(np.nanmean(e[m], 0)))
NEST = np.full_like(H, np.nan); picks = {}
for f in range(5):
    tr, te = fold != f, fold == f; best = min(CFG, key=lambda n: off(err[n], tr)); NEST[te] = PP[best][te]; picks[f] = (best, round(off(err[best], tr) - off(E.ape(REF, y), tr), 4))
print('picked per fold (name, delta on the other four)', json.dumps(picks), flush=True); out['nested'] = rep('nested selection over 22 configurations', NEST, REF); out['after_looking'] = rep('pre x3, picked after looking', PP['pre x3'], REF)
d = {n: off(err[n], np.ones(N_, bool)) - off(E.ape(REF, y), np.ones(N_, bool)) for n in CFG}; print('all configs, delta on all folds', json.dumps({k: round(v, 4) for k, v in sorted(d.items(), key=lambda kv: kv[1])}), flush=True)
print('=== B transfer to other bases', flush=True); pr = load(PREX3)
seeds, single, hist = C.load_members(b); bases = {'recipe A two seeds': b['A'], 'crop five': b['P1'], 'seed 46 TTA': seeds[46], 'seed 47 TTA': seeds[47], 'hybrid B': B}
for nm, base in bases.items():
    H_ = hex6(base); out['base ' + nm] = rep('graph on ' + nm, pipe(H_, pr), pipe(H_))
SIX = 'raw_ssl_next_20260913/ssl_runs/fold{f}/standard/oof.npz'; p6 = np.full_like(B, np.nan)
for f in (2, 3, 4):
    five = []
    for n, pat in C.HIST.items():
        with np.load(C.X / pat.format(f=f), allow_pickle=True) as z: five.append((np.array([pos[int(i)] for i in z['indices']]), z['pred'].astype(np.float64)))
    with np.load(C.X / SIX.format(f=f), allow_pickle=True) as z: r6 = np.array([pos[int(i)] for i in z['indices']]); q6 = z['pred'].astype(np.float64)
    a6 = np.zeros((N_, 3)); c6 = np.zeros(N_)
    for r, q in five + [(r6, q6)]: a6[r] += q; c6[r] += 1
    ok = c6 == 6; p6[ok] = (a6[ok] / 6) * C.SCALE
m6 = np.isfinite(p6).all(1); B6 = np.where(m6[:, None], 0.5 * b['A'] + 0.5 * p6, B); H6 = hex6(B6); out['base deployed six'] = rep('graph on the deployed crop six, folds 2-4, stretch fit on these rows', pipe(H6, pr, rows=m6), pipe(H6, rows=m6), m6)
print('=== C what it learned', flush=True); Oof = np.zeros((N_, 3))
for f in range(5): Oof[fold == f] = pr[f][fold == f]
r = ly - np.log(REF); rH = ly - np.log(H)
def cor(a, c):
    s = np.isfinite(a) & np.isfinite(c); return round(float(np.corrcoef(a[s], c[s])[0, 1]), 3)
z = np.load(O / 'window_multi.npz'); Fw = z['F'][idx].astype(np.float64); fam = [str(x) for x in z['families']]; st = [str(x) for x in z['stats']]; P0 = Fw[:, fam.index('preonly'), 0]; lg = lambda v: np.where(v > 0, np.log(np.where(v > 0, v, 1)), np.nan)
for j, k in ((1, 'CV'), (2, 'HEX'), (0, 'CD')):
    s = np.isfinite(rH[:, j]); be = float((Oof[s, j] * rH[s, j]).sum() / (Oof[s, j] ** 2).sum()); print(k, 'pred SD', round(float(Oof[:, j].std()), 4), 'residual SD', round(float(np.nanstd(rH[:, j])), 4), 'r with residual', cor(Oof[:, j], rH[:, j]), 'slope of residual on pred', round(be, 3), 'pred mean', round(float(Oof[:, j].mean()), 4), flush=True)
    print('   r with: window hex6', cor(Oof[:, j], lg(P0[:, st.index('hex6')])), 'window CV dist', cor(Oof[:, j], lg(P0[:, st.index('cv_dist')])), 'window cells', cor(Oof[:, j], lg(P0[:, st.index('n')])), 'window CD', cor(Oof[:, j], lg(P0[:, st.index('cd')])), 'ten-detector hex6', cor(Oof[:, j], f6), '| log B CD, CV, HEX', cor(Oof[:, j], np.log(H[:, 0])), cor(Oof[:, j], np.log(H[:, 1])), cor(Oof[:, j], np.log(H[:, 2])), flush=True)
P_ = PP['pre x3']; e1, e0 = E.ape(P_, y), E.ape(REF, y); dm = np.nanmean(e1 - e0, 1)
def bins(nm, v, q=(0, .1, .25, .5, .75, .9, 1)):
    ed = np.nanquantile(v, q); print(nm, [(round(float(ed[i]), 3), round(float(np.nanmean(dm[(v >= ed[i]) & (v <= ed[i + 1])])), 3)) for i in range(len(q) - 1)], flush=True)
bins('gain by HEX correction', Oof[:, 2]); bins('gain by size of HEX correction', np.abs(Oof[:, 2])); bins('gain by CV correction', Oof[:, 1]); bins('gain by true log HEX (not usable at test)', ly[:, 2]); bins('gain by B log HEX', np.log(H[:, 2])); bins('gain by B log CD', np.log(H[:, 0])); bins('gain by window cells', lg(P0[:, st.index('n')]))
for f in range(5):
    m = fold == f; print('fold', f, 'residual SD HEX', round(float(np.nanstd(rH[m, 2])), 4), 'r pred-resid HEX', cor(Oof[m, 2], rH[m, 2]), 'CV', cor(Oof[m, 1], rH[m, 1]), 'REF', round(E.official(E.ape(REF[m], y[m]))['mean'], 3), 'new-format share', round(float((fmt[m] == 1).mean()), 3), flush=True)
json.dump(out, open(O / 'graph_review.json', 'w'), indent=1, default=float); print('done', flush=True)
