"""Strict read of a member that exists on some folds only (e.g. folds 0 and 4) inside the 39-column joint ridge (HEX lane, 2026-11-11).
The member enters as its log HEX deviation from the A/P1 blend on its folds and 0 elsewhere (no flag: it would act as a fold indicator). For outer
fold f the joint procedure is rerun with fold f's labels hidden; variants: the 39-column ridge and the same plus the member.
usage: strict_part_member.py FOLD OUTDIR ROOT ARCH FOLDS EPOCHS   (ROOT with {f}; FOLDS e.g. 0,4; EPOCHS e.g. 8-12)
       strict_part_member.py assemble OUTDIR"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')

if sys.argv[1] == 'assemble':
    import chain as C
    y, fold, ok = C.y[:, 2], C.fold, C.ok; Z = [np.load(f'{sys.argv[2]}/spm_f{f}.npz') for f in range(5)]; out = {}
    for v in ('base39', 'member'):
        P = C.final({f: Z[f][v] for f in range(5)})[:, 2]; E = np.abs(P - y) / np.where(ok, y, 1)
        out[v] = dict(all5=round(float(E[ok].mean() * 100), 5), f04=round(float(E[ok & np.isin(fold, [0, 4])].mean() * 100), 5), folds=[round(float(E[ok & (fold == f)].mean() * 100), 4) for f in range(5)])
    out['delta'] = dict(all5=round(out['member']['all5'] - out['base39']['all5'], 5), f04=round(out['member']['f04'] - out['base39']['f04'], 5), folds=[round(a - b, 4) for a, b in zip(out['member']['folds'], out['base39']['folds'])])
    out['weights'] = [json.load(open(f'{sys.argv[2]}/spm_f{f}.json')) for f in range(5)]
    print(json.dumps(out, indent=1)); json.dump(out, open(f'{sys.argv[2]}/strict_part_member.json', 'w'), indent=1); sys.exit(0)

F_OUT, OUT, ROOT, ARCH = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
FS = [int(q) for q in sys.argv[5].split(',')]; e0, e1 = [int(q) for q in sys.argv[6].split('-')]
import hexlib as L
L.y[L.fold == F_OUT, :] = 0.0; L.good[L.fold == F_OUT] = False
import chain as C
import joint_fit as JF
JF.ok, JF.good, JF.y = C.ok, C.good, C.y[:, 2]
G = JF.columns(); names = [k for k in G if k != 'gnn4647']; X39 = np.concatenate([G[k] for k in names], 1)
lm = np.log(C.part_hex({f: ROOT.format(f=f) for f in FS}, ARCH, range(e0, e1 + 1))); A, P1 = C.BP._b['A'][:, 2], C.BP._b['P1'][:, 2]
av = np.isfinite(lm); dev = np.where(av, lm - (0.7 * np.log(P1) + 0.3 * np.log(A)), 0.0)
X = {'base39': X39, 'member': np.c_[X39, dev]}   # no availability flag: with only folds 0 and 4 covered it acts as a fold indicator and shifts the other folds
bases = {'hb': {f: C.L.HB[f].copy() for f in range(5)}, 'ap1_nested': C.STEPS['ap1']({f: C.L.HB[f].copy() for f in range(5)})[0]}
res, info = {}, dict(fold=F_OUT, rows_with_member=int(av.sum()))
for v, XX in X.items():
    outs = {}
    for bn, b in bases.items(): JF.INNER.clear(); r, cd = JF.run(b, XX); outs[bn] = (cd, list(JF.INNER.values())[0][F_OUT], r['alpha_picks'][F_OUT])
    pk = min(outs, key=lambda k: outs[k][1]); res[v] = outs[pk][0][F_OUT]; info[v + '_base'] = pk
    if v == 'member':   # standardized weight of the member column in this outer fold's fit
        base = bases[pk][F_OUT]; t = JF.lg(np.where(C.ok, C.y[:, 2], 1)) - np.log(base); Xf = np.c_[XX, np.log(base)]
        med = np.nanmedian(np.where(np.isfinite(Xf), Xf, np.nan)[(C.fold != F_OUT) & C.good], 0); Xf = np.where(np.isfinite(Xf), Xf, med); tr = (C.fold != F_OUT) & C.good & C.ok
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9; A_ = ((Xf - mu) / sd)[tr]; tt = t[tr]
        w = np.linalg.solve(A_.T @ A_ + outs[pk][2] * np.eye(A_.shape[1]), A_.T @ (tt - tt.mean())); info['member_weight_std'] = round(float(w[X39.shape[1]]), 4)
np.savez_compressed(f'{OUT}/spm_f{F_OUT}.npz', **res); json.dump(info, open(f'{OUT}/spm_f{F_OUT}.json', 'w')); print('done', F_OUT, flush=True)
