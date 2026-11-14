"""Strict five-fold read of extra columns in the joint ridge (HEX lane, 2026-11-11): for outer fold f the joint-ridge procedure
(base choice, alpha by inner leave-one-fold-out, fit, stretch) is rerun with fold f's labels hidden; variants: the 39-column freeze
ridge, and the same plus the backbone lane's synthetic-pretrained zero-shot columns (log CD, log area CV, log HEX).
usage: strict_joint_extra.py FOLD OUTDIR   |   strict_joint_extra.py assemble OUTDIR"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
SYN = '/workspace/clear-ec-research/experiments/lane_backbone_20261108/data/synth_zeroshot_f0pre_20261111.npz'
VARIANTS = ('base39', 'synth3')

if sys.argv[1] == 'assemble':
    import chain as C
    y, fold, ok = C.y[:, 2], C.fold, C.ok; Z = [np.load(f'{sys.argv[2]}/sje_f{f}.npz') for f in range(5)]; out = {}
    for v in VARIANTS:
        P = C.final({f: Z[f][v] for f in range(5)})[:, 2]; E = np.abs(P - y) / np.where(ok, y, 1)
        out[v] = dict(all5=round(float(E[ok].mean() * 100), 5), f04=round(float(E[ok & np.isin(fold, [0, 4])].mean() * 100), 5), folds=[round(float(E[ok & (fold == f)].mean() * 100), 4) for f in range(5)])
    out['delta_synth3_vs_base39'] = dict(all5=round(out['synth3']['all5'] - out['base39']['all5'], 5), folds=[round(a - b, 4) for a, b in zip(out['synth3']['folds'], out['base39']['folds'])])
    out['bases'] = [json.load(open(f'{sys.argv[2]}/sje_f{f}.json')) for f in range(5)]
    print(json.dumps(out, indent=1)); json.dump(out, open(f'{sys.argv[2]}/strict_joint_extra.json', 'w'), indent=1); sys.exit(0)

F_OUT = int(sys.argv[1]); OUT = sys.argv[2]
import hexlib as L
L.y[L.fold == F_OUT, :] = 0.0; L.good[L.fold == F_OUT] = False
import chain as C
assert not C.ok[C.fold == F_OUT].any()
import joint_fit as JF
JF.ok, JF.good, JF.y = C.ok, C.good, C.y[:, 2]
G = JF.columns(); names = [k for k in G if k != 'gnn4647']; X39 = np.concatenate([G[k] for k in names], 1)
z = np.load(SYN); pos = {int(i): k for k, i in enumerate(z['idx'])}; S = np.log(z['pred'][[pos[int(i)] for i in C.idx]].astype(float))
X = {'base39': X39, 'synth3': np.c_[X39, S]}
bases = {'hb': {f: C.L.HB[f].copy() for f in range(5)}, 'ap1_nested': C.STEPS['ap1']({f: C.L.HB[f].copy() for f in range(5)})[0]}
res, pick = {}, {}
for v, XX in X.items():
    outs = {}
    for bn, b in bases.items(): JF.INNER.clear(); r, cd = JF.run(b, XX); outs[bn] = (cd, list(JF.INNER.values())[0][F_OUT])
    pick[v] = min(outs, key=lambda k: outs[k][1]); res[v] = outs[pick[v]][0][F_OUT]
np.savez_compressed(f'{OUT}/sje_f{F_OUT}.npz', **res); json.dump(dict(fold=F_OUT, base=pick), open(f'{OUT}/sje_f{F_OUT}.json', 'w')); print('done', F_OUT, flush=True)
