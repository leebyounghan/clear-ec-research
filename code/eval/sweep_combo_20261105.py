"""Model-combination sweep on stored OOF predictions (2026-11-05). All selection is nested: for each held fold, member scalars, weights and the choice of members come from the other four folds only, and the held fold is read once. Against hybrid B built the same way, paired donor-cluster bootstrap.
  ablation     B with one member dropped; B with one library member added at 20 %
  family       recipe A vs Phase-1 weight per metric
  greedy       forward selection with replacement, whole-score or per metric, from empty or starting from B, plain or bagged over row resamples
  nnls         non-negative weights on log predictions per metric
  geometric    B with a geometric instead of arithmetic mean
Best nested candidates then go through hex6 + stretch. P1-type members were chosen by validation score per fold, recipe A members are terminal: this favours P1 in any selection. Forked pool, clamp(16 - load1 - 1, 2, 12)."""
import json, os, re, sys
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.optimize import nnls
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
E = C.E; K = list(C.K)
OUT = C.X / 'meta_ceiling_20261103/sweep'; b = E.load_base(); y, fold, Bs = b['y'], b['fold'], b['B']; g = E.donors(b['ids']); ok3 = y > 0
z = np.load(OUT / 'library.npz', allow_pickle=True); names = [str(n) for n in z['names']]; comp = z['complete']; ALL = z['preds'].astype(np.float64); P = ALL[:, comp]; NAMES = [n for n, c in zip(names, comp) if c]; M_ = len(NAMES)
print('complete members', M_, flush=True)
BNAMES = dict(s46='recipeA_s46_tta', s47='recipeA_s47_tta', metric35='methods_20260904/metric35_f', predbag='methods_20260904/predbag2_35_f_seed42_c256k2_mean', s77='submission_20260904/base_seed77_f_seed77_c256k1_mean', s2026='submission_20260904/base_seed2026_f_seed2026_c256k1_mean', s42='followup_20260902/fixed_base_f')
Bix = {k: NAMES.index(v) for k, v in BNAMES.items()}; BW = np.zeros(M_); BW[Bix['s46']] = BW[Bix['s47']] = 0.25
for k in ('metric35', 'predbag', 's77', 's2026', 's42'): BW[Bix[k]] = 0.1


def wmed(p, t):
    r = t / p; w = p / t; o = np.argsort(r); c = np.cumsum(w[o]); return r[o][np.searchsorted(c, c[-1] / 2)]


def member_scalars(tr, X=None):
    X = P if X is None else X; S = np.ones((X.shape[1], 3))
    for m in range(X.shape[1]):
        for j in range(3):
            q = tr & ok3[:, j] & np.isfinite(X[:, m, j]); S[m, j] = wmed(X[q, m, j], y[q, j]) if q.sum() > 50 else 1.0
    return S


SCALED = {f: P * member_scalars(fold != f)[None] for f in range(5)}


def final(pred, f):
    """Scalar on the combination from the other folds; returns the held fold's rows."""
    tr, te = fold != f, fold == f; out = pred[te].copy()
    for j in range(3):
        q = tr & ok3[:, j]; out[:, j] *= wmed(pred[q, j], y[q, j])
    return out


def assemble(fn):
    out = np.full_like(Bs, np.nan); info = {}
    for f in range(5):
        p, i = fn(f); out[fold == f] = final(p, f); info[f] = i
    return out, info


def ref_B(f): return (SCALED[f] * BW[None, :, None]).sum(1) / BW.sum(), None
REF, _ = assemble(ref_B)


def report(name, p, info=None, ref=REF):
    pb = E.paired_bootstrap(E.ape(p, y), E.ape(ref, y), g); s = E.official(E.ape(p, y)); s0 = E.official(E.ape(ref, y))
    per = [round(E.official(E.ape(p[fold == f], y[fold == f]))['mean'] - E.official(E.ape(ref[fold == f], y[fold == f]))['mean'], 3) for f in range(5)]
    return dict(name=name, score=round(s['mean'], 3), per={k: round(s[k], 3) for k in K}, per_delta={k: round(s[k] - s0[k], 3) for k in K}, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per, info=info)


def combo(w, X): return (X * w[None, :, None]).sum(1) / w.sum()


def greedy(X, tr, metrics, steps, init=None, rows=None):
    """Caruana forward selection with replacement on the training rows. metrics: tuple of metric indices scored together. Returns weights."""
    rows = np.where(tr)[0] if rows is None else rows; w = np.zeros(M_) if init is None else init.copy() * (20 if init is not None else 1); cur = (X[rows] * w[None, :, None]).sum(1); tot = w.sum()
    Xr = X[rows]; yr = y[rows]; s = ok3[rows]; best = (9e9, w.copy())
    for k in range(steps):
        sc = np.zeros(M_)
        for j in metrics:
            q = s[:, j]; e = np.abs((cur[q, j][:, None] + Xr[q, :, j]) / (tot + 1) - yr[q, j][:, None]) / yr[q, j][:, None]; sc += e.mean(0)
        m = int(np.argmin(sc)); w[m] += 1; cur += Xr[:, m]; tot += 1
        if sc[m] < best[0] - 1e-6: best = (float(sc[m]), w.copy())
    return best[1]


def job(a):
    kind, f, arg = a; X = SCALED[f]; tr = fold != f
    if kind == 'greedy_all':
        w = greedy(X, tr, (0, 1, 2), arg['steps'], arg.get('init')); return f, [w, w, w]
    if kind == 'greedy_metric':
        return f, [greedy(X, tr, (j,), arg['steps'], arg.get('init')) for j in range(3)]
    if kind == 'bag':
        rng = np.random.default_rng(100 + f); W = [np.zeros(M_) for _ in range(3)]; rows0 = np.where(tr)[0]; pool_ids = np.unique(g[rows0])
        for r in range(arg['bags']):
            pick = rng.choice(pool_ids, len(pool_ids)); mask = np.isin(g[rows0], pick); rows = rows0[mask]
            for j in range(3):
                v = greedy(X, tr, (j,), arg['steps'], arg.get('init'), rows); W[j] += v / v.sum()
        return f, W
    if kind == 'nnls':
        W = []
        for j in range(3):
            q = tr & ok3[:, j]; A = np.log(X[q, :, j]); t = np.log(y[q, j]); ma = A.mean(0); v, _ = nnls(A - ma, t - t.mean()); W.append(v if v.sum() > 0 else BW.copy())
        return f, W
    raise ValueError(kind)


def run(pool, name, kind, **arg):
    res = dict(pool.map(job, [(kind, f, arg) for f in range(5)]))
    def fn(f):
        X = SCALED[f]; return np.stack([combo(res[f][j], X)[:, j] for j in range(3)], 1), [{NAMES[m][:44]: round(float(res[f][j][m] / res[f][j].sum()), 3) for m in np.argsort(-res[f][j])[:6] if res[f][j][m] > 0} for j in range(3)]
    p, info = assemble(fn); return report(name, p, info), p


def main():
    workers = int(min(5, max(2, 16 - os.getloadavg()[0] - 1))); rep = {}; preds = {}
    rep['ref'] = report('B rebuilt, vs itself', REF); s0 = report('stored B vs rebuilt', Bs); print('rebuilt B', rep['ref']['score'], rep['ref']['per'], 'stored B minus rebuilt', s0['delta'], flush=True)
    # geometric
    p, _ = assemble(lambda f: (np.exp((np.log(SCALED[f]) * BW[None, :, None]).sum(1) / BW.sum()), None)); rep['geometric'] = report('B geometric', p); print(json.dumps(rep['geometric']), flush=True)
    # drop one
    for k, m in Bix.items():
        w = BW.copy(); w[m] = 0
        if k in ('s46', 's47'): w[Bix['s46']] = w[Bix['s47']] = 0; w[m] = 0; o = Bix['s47'] if k == 's46' else Bix['s46']; w[o] = 0.5
        else: w[[Bix[q] for q in ('metric35', 'predbag', 's77', 's2026', 's42') if q != k]] = 0.125
        p, _ = assemble(lambda f: (combo(w, SCALED[f]), None)); rep[f'drop_{k}'] = report(f'drop {k}', p); print('drop', k, rep[f'drop_{k}']['delta'], rep[f'drop_{k}']['ci'], flush=True)
    # add one at 20 %
    add = []
    for m in range(M_):
        if BW[m] > 0: continue
        w = BW.copy() * 0.8; w[m] = 0.2; p, _ = assemble(lambda f: (combo(w, SCALED[f]), None)); r = report(NAMES[m], p); add.append(r)
    add.sort(key=lambda r: r['delta']); rep['add_one'] = add; print('add one at 20 %, best', json.dumps([{k: r[k] for k in ('name', 'score', 'delta', 'ci', 'folds')} for r in add[:8]]), flush=True); print(' worst', [(r['name'][:30], r['delta']) for r in add[-3:]], flush=True)
    # family weight per metric
    def fam(f):
        X = SCALED[f]; A_ = combo((BW * np.isin(np.arange(M_), [Bix['s46'], Bix['s47']])), X); C_ = combo((BW * np.isin(np.arange(M_), [Bix[q] for q in ('metric35', 'predbag', 's77', 's2026', 's42')])), X); o = np.zeros_like(A_); ws = []
        for j in range(3):
            q = (fold != f) & ok3[:, j]; gr = np.linspace(0, 1, 21); e = [np.mean(np.abs(w_ * A_[q, j] + (1 - w_) * C_[q, j] - y[q, j]) / y[q, j]) for w_ in gr]; w_ = gr[int(np.argmin(e))]; o[:, j] = w_ * A_[:, j] + (1 - w_) * C_[:, j]; ws.append(float(w_))
        return o, ws
    p, info = assemble(fam); rep['family_weight'] = report('family weight per metric', p, info); print(json.dumps({k: rep['family_weight'][k] for k in ('score', 'delta', 'ci', 'per_delta', 'folds', 'info')}), flush=True)
    with get_context('fork').Pool(workers) as pool:
        for nm, kind, arg in (('greedy whole score', 'greedy_all', dict(steps=40)), ('greedy per metric', 'greedy_metric', dict(steps=40)), ('greedy whole, from B', 'greedy_all', dict(steps=20, init=BW)), ('greedy per metric, from B', 'greedy_metric', dict(steps=20, init=BW)), ('bagged greedy per metric', 'bag', dict(steps=25, bags=20)), ('bagged per metric, from B', 'bag', dict(steps=10, bags=20, init=BW)), ('nnls per metric', 'nnls', {})):
            r, p = run(pool, nm, kind, **arg); rep[nm] = r; preds[nm] = p; print(json.dumps({k: r[k] for k in ('name', 'score', 'per', 'per_delta', 'delta', 'ci', 'folds')}), flush=True); print('   fold 0 picks', json.dumps(r['info'][0]), flush=True)
    # hex6 + stretch on top
    A_, B_, W_ = -0.2084, 0.5989, 0.28; gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']
    def post(base):
        H = base.copy(); H[have, 2] = np.exp((1 - W_) * np.log(base[have, 2]) + W_ * (A_ + B_ * gf['f_active10'][have])); out = np.full_like(H, np.nan)
        for f in range(5):
            s = N.fit_stretch(H, y, (fold != f) & good); out[fold == f] = N.apply_stretch(H[fold == f], s)
        return out
    PR = post(REF); rep['post_ref'] = report('B + hex6 + stretch vs B', PR); print(json.dumps({k: rep['post_ref'][k] for k in ('score', 'per', 'delta', 'ci')}), flush=True)
    top = sorted(preds, key=lambda n: rep[n]['delta'])[:3]
    for n in top:
        rep['post_' + n] = report(n + ' + hex6 + stretch, vs B + hex6 + stretch', post(preds[n]), None, PR); print(json.dumps({k: rep['post_' + n][k] for k in ('name', 'score', 'per', 'delta', 'ci', 'folds')}), flush=True)
    # partial families, added at 20 % on their own rows
    Bfull = REF; part = []
    for i, n in enumerate(names):
        if comp[i]: continue
        a = ALL[:, i]; s = np.isfinite(a).all(1)
        if s.sum() < 1700: continue
        sc = a.copy(); fs = np.unique(fold[s]); 
        for f in fs:
            tr = (fold != f) & s
            if tr.sum() < 500: tr = s & (fold == f)                      # single-fold family: in-sample scalar, a slight optimism flagged below
            for j in range(3):
                q = tr & ok3[:, j]; sc[fold == f, j] *= wmed(a[q, j], y[q, j])
        p = Bfull.copy(); p[s] = 0.8 * Bfull[s] + 0.2 * sc[s]; pb = E.paired_bootstrap(E.ape(p[s], y[s]), E.ape(Bfull[s], y[s]), np.unique(g[s], return_inverse=True)[1]); part.append(dict(name=n, rows=int(s.sum()), folds=fs.tolist(), alone=round(E.official(E.ape(a[s], y[s]))['mean'], 3), delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']]))
    part.sort(key=lambda r: r['delta']); rep['partial_add'] = part; print('partial families, best added at 20 %', flush=True)
    for r in part[:18]: print(' ', json.dumps(r), flush=True)
    json.dump(rep, open(OUT / 'combo_sweep.json', 'w'), indent=1, default=float); np.savez_compressed(OUT / 'combo_preds.npz', **{re.sub(r'\W+', '_', k): v for k, v in preds.items()}, ref=REF); print('done', flush=True)


if __name__ == '__main__':
    main()
