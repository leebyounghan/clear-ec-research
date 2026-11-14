"""Calibrated read of RoPE scale: the same stretch on both arms, fit on the other fold, so a plain shift in level does not count as a gain."""
import sys, numpy as np
sys.path[:0] = ['experiments/meta_ceiling_20261103/source', 'experiments/eval_harness_20261031/source']
import eval_oof_20261031 as E, stretch_nested_20261103 as N
base = E.load_base(); gp = E.donors(base['ids']); pos = {int(i): n for n, i in enumerate(base['idx'])}
Z = {f: np.load(f'experiments/lane_gpu0_20261106/rope_tta/f{f}_s46.npz') for f in (0, 4)}
y = {f: Z[f]['target'].astype(np.float64) for f in Z}
g = np.concatenate([gp[[pos[int(i)] for i in Z[f]['indices']]] for f in Z])
keys = sorted(k for k in Z[0].files if k.startswith('scale'))
print(keys)
def cal(fn):
    """fn(f) gives predictions for fold f. Returns stacked predictions after the stretch fit on the other fold."""
    out = []
    for f, o in ((0, 4), (4, 0)):
        o_p = fn(o); st = N.fit_stretch(o_p, y[o], np.ones(len(o_p), bool)); out.append(N.apply_stretch(fn(f), st))
    return np.concatenate(out)
yy = np.concatenate([y[0], y[4]])
pure = lambda k: (lambda f: Z[f][k].astype(np.float64))
avg = lambda ks: (lambda f: np.exp(np.mean([np.log(Z[f][k].astype(np.float64)) for k in ks], 0)))
ref_raw = np.concatenate([Z[f]['scale_1.000'].astype(np.float64) for f in (0, 4)])
ref = cal(pure('scale_1.000')); s0 = E.official(E.ape(ref, yy)); print('scale 1 raw', round(E.official(E.ape(ref_raw, yy))['mean'], 4), 'calibrated', round(s0['mean'], 4), {k: round(s0[k], 3) for k in ('CD', 'CV', 'HEX')})
for name, fn in (('scale 0.707', pure('scale_0.707')), ('scale 1.414', pure('scale_1.414')), ('average 0.707 1 1.414', avg(keys)), ('average 1 1.414', avg(['scale_1.000', 'scale_1.414']))):
    p = cal(fn); pb = E.paired_bootstrap(E.ape(p, yy), E.ape(ref, yy), g); s = E.official(E.ape(p, yy))
    n = len(Z[0]['indices']); per = [round(E.official(E.ape(p[:n], y[0]))['mean'] - E.official(E.ape(ref[:n], y[0]))['mean'], 4), round(E.official(E.ape(p[n:], y[4]))['mean'] - E.official(E.ape(ref[n:], y[4]))['mean'], 4)]
    print(name.ljust(24), 'mean', round(s['mean'], 4), 'delta', round(pb['delta'], 4), 'CI', np.round(pb['ci95'], 4).tolist(), {k: round(s[k] - s0[k], 4) for k in ('CD', 'CV', 'HEX')}, 'folds', per)
