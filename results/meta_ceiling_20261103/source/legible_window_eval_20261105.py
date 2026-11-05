"""Read the legible-window statistics against the labels on all frames (2026-11-05). r with log y and with the log residual of B + hex6 + stretch; partial after the shipped hex6; LOFO log blend per metric inside the deployed pipeline, B -> hex6 -> window blend -> stretch. Gate written before reading: mean <= -0.02 on top of B + hex6 + stretch, CI upper bound < 0, at least 4 of 5 folds negative. Also on the 25: window statistic vs device value."""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); OUT = C.X / 'meta_ceiling_20261103/sweep'; b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); ids = [str(i).strip() for i in b['ids']]; N_ = len(y)
z = np.load(OUT / 'legible_window.npz'); names = [str(n) for n in z['names']]; F = z['F'][idx].astype(np.float64); get = lambda n: F[:, names.index(n)]
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float)
H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
def pipeline(base):
    out = np.full_like(base, np.nan)
    for f in range(5): out[fold == f] = N.apply_stretch(base[fold == f], N.fit_stretch(base, y, (fold != f) & good))
    return out
REF = pipeline(H); ly = np.log(np.where(y > 0, y, np.nan)); r = ly - np.log(REF); rB = ly - np.log(B)
def cor(a, c):
    s = np.isfinite(a) & np.isfinite(c); return round(float(np.corrcoef(a[s], c[s])[0, 1]), 3)
def res(a, x):
    s = np.isfinite(a) & np.isfinite(x); o = np.full(len(a), np.nan); A = np.c_[np.ones(s.sum()), x[s]]; o[s] = a[s] - A @ np.linalg.lstsq(A, a[s], rcond=None)[0]; return o
print('frames', int(np.isfinite(get('top_hull_hex6')).sum()), 'B + hex6 + stretch', round(E.official(E.ape(REF, y))['mean'], 3), 'n =', N_, 'SE ~', round(1 / np.sqrt(N_), 3)); print('hex6 ten-detector: r with log HEX', cor(f6, ly[:, 2]), 'with B residual', cor(f6, rB[:, 2]), 'with log CV', cor(f6, ly[:, 1]))
rep = {}
for rule in ('whole', 'centre_hull', 'centre_box', 'top_hull', 'top_box', 'top3_hull', 'top3_box'):
    o = {}
    for s_, j in (('hex6', 2), ('hex6', 1), ('cv_area', 1), ('cv_dist', 1), ('cd', 0)):
        v = get(f'{rule}_{s_}'); v = np.where(v > 0, np.log(v), np.nan); o[f'{s_}->{K[j]}'] = (cor(v, ly[:, j]), cor(v, rB[:, j]), cor(v, r[:, j]), cor(res(v, f6), res(r[:, j], f6)))
    rep[rule] = o; print(rule, '[r y, r B-resid, r final-resid, partial after hex6]', json.dumps(o), flush=True)
def blend(base, v, j, tr):
    """(1 - w) log p + w (a + b log v), a, b, w on training rows"""
    q = tr & np.isfinite(v) & (y[:, j] > 0); A = np.polyfit(v[q], np.log(y[q, j]), 1); best = (9e9, 0.0)
    for w in np.arange(0, 0.71, 0.02):
        e = np.mean(np.abs(np.exp((1 - w) * np.log(base[q, j]) + w * np.polyval(A, v[q])) - y[q, j]) / y[q, j])
        if e < best[0] - 1e-7: best = (e, w)
    return A, best[1]
def run(spec):
    P = np.full_like(B, np.nan); ws = []
    for f in range(5):
        tr = (fold != f) & good; C_ = H.copy(); W = {}
        for name, j in spec:
            v = get(name); v = np.where(v > 0, np.log(v), np.nan); A, w = blend(C_, v, j, tr); s = np.isfinite(v); C_[s, j] = np.exp((1 - w) * np.log(C_[s, j]) + w * np.polyval(A, v[s])); W[f'{name}->{K[j]}'] = round(float(w), 2)
        P[fold == f] = N.apply_stretch(C_[fold == f], N.fit_stretch(C_, y, tr)); ws.append(W)
    return P, ws
def report(nm, spec):
    p, ws = run(spec); pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); s, s0 = E.official(E.ape(p, y)), E.official(E.ape(REF, y)); per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(REF[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]; o = dict(score=round(s['mean'], 3), per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per, w0=ws[0]); print(nm, json.dumps(o), flush=True); return o
st = {}
for rule in ('top_hull', 'top_box', 'top3_hull', 'top3_box', 'centre_hull', 'whole'):
    st[rule + ' HEX'] = report(rule + ' hex6 on HEX', [(f'{rule}_hex6', 2)]); st[rule + ' CV'] = report(rule + ' cv_dist on CV', [(f'{rule}_cv_dist', 1)]); st[rule + ' CD'] = report(rule + ' cd on CD', [(f'{rule}_cd', 0)])
json.dump(dict(corr=rep, stack=st), open(OUT / 'legible_window_eval.json', 'w'), indent=1); print('done', flush=True)
