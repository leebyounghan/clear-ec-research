"""Fix the stretch parameters for the final package (2026-11-04). Pipeline: hybrid B -> deployed hex6 (a, b, w frozen) on HEX ->
per-metric stretch p' = c exp(g (log p - mu) + mu), g on a 0.025 grid, c the MAPE-optimal weighted median, fit on all OOF rows
except the 20 annotated training rows seen by the detectors. Then checks that matter for deployment: LOFO repeat with the same
grid; slope sensitivity (+-0.025 around the chosen g); transfer to the deployed composition, Phase-1 six instead of five, which
exists on folds 2, 3, 4 only (stretch fit on the five-member rows of the other folds, scored on the six-member rows)."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = C.K
A_, B_, W_ = -0.2084, 0.5989, 0.28
b = E.load_base(); y, B, fold = b['y'], b['B'], b['fold']; pos = {int(i): k for k, i in enumerate(b['idx'])}
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True)
have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10'])
def hex6(base):
    h = base.copy(); h[have, 2] = np.exp((1 - W_) * np.log(base[have, 2]) + W_ * (A_ + B_ * gf['f_active10'][have])); return h
H = hex6(B); g = E.donors(b['ids']); rows = ~gf['seen_leaked']
prm = N.fit_stretch(H, y, rows)
final = {K[j]: dict(g=float(x[0]), c=float(x[1]), mu=float(x[2])) for j, x in enumerate(prm)}
out = dict(params=final, hex6=dict(a=A_, b=B_, w=W_), fit_rows=int(rows.sum()))
P = N.apply_stretch(H, prm)
def rep(name, p, ref, m=None):
    m = np.ones(len(y), bool) if m is None else m
    pb = E.paired_bootstrap(E.ape(p[m], y[m]), E.ape(ref[m], y[m]), np.unique(g[m], return_inverse=True)[1])
    out[name] = dict(score=C.mape(p[m], y[m]), ref=C.mape(ref[m], y[m]), delta=pb['delta'], ci95=pb['ci95'])
    print(name, {k: round(v, 3) for k, v in out[name]['score'].items() if k != 'n'}, round(pb['delta'], 4), [round(v, 4) for v in pb['ci95']], flush=True)
rep('in_sample_hex6_stretch_vs_B', P, B); rep('hex6_vs_B', H, B)
Pl = np.full_like(B, np.nan)
for f in range(5):
    s = N.fit_stretch(H, y, (fold != f) & rows); Pl[fold == f] = N.apply_stretch(H[fold == f], s)
    out.setdefault('lofo_slopes', {})[f] = [float(x[0]) for x in s]
rep('lofo_hex6_stretch_vs_B', Pl, B)
for d in (-0.025, 0.025):
    alt = [(g_ + d, c_, mu_) for g_, c_, mu_ in prm]
    rep(f'slope{d:+.3f}_vs_B', N.apply_stretch(H, alt), B)
# Phase-1 six, deployed composition, folds 2-4
SIX = 'raw_ssl_next_20260913/ssl_runs/fold{f}/standard/oof.npz'
p6 = np.full_like(B, np.nan)
for f in (2, 3, 4):
    five = []
    for n, pat in C.HIST.items():
        with np.load(C.X / pat.format(f=f), allow_pickle=True) as z:
            five.append((np.array([pos[int(i)] for i in z['indices']]), z['pred'].astype(np.float64)))
    with np.load(C.X / SIX.format(f=f), allow_pickle=True) as z:
        r6 = np.array([pos[int(i)] for i in z['indices']]); q6 = z['pred'].astype(np.float64)
    a6 = np.zeros((len(pos), 3)); c6 = np.zeros(len(pos))
    for r, q in five + [(r6, q6)]:
        a6[r] += q; c6[r] += 1
    ok = c6 == 6; p6[ok] = (a6[ok] / 6) * C.SCALE
m6 = np.isfinite(p6).all(1)
B6 = np.where(m6[:, None], 0.5 * b['A'] + 0.5 * p6, B); H6 = hex6(B6)
Ps = np.full_like(B, np.nan)
for f in (2, 3, 4):
    s = N.fit_stretch(H, y, (fold != f) & rows)
    te = (fold == f) & m6; Ps[te] = N.apply_stretch(H6[te], s)
out['six_member_rows'] = int(m6.sum())
rep('six_rows_hex6_vs_B6', H6, B6, m6); rep('six_rows_hex6_stretch_vs_B6', np.where(m6[:, None], Ps, B6), B6, m6); rep('six_rows_stretch_only_gain', np.where(m6[:, None], Ps, B6), H6, m6)
out['spread_ratio_six_over_five'] = {k: float(np.log(B6[m6, j]).std() / np.log(B[m6, j]).std()) for j, k in enumerate(K)}
print('params', json.dumps(final)); print('lofo slopes', out['lofo_slopes']); print('spread six/five', out['spread_ratio_six_over_five'])
json.dump(out, open(C.OUT / 'stretch_final.json', 'w'), indent=1, default=float)
