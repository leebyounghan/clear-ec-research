"""Strict price ladder of the HEX packaging tiers (HEX lane, 2026-11-12, D-PKG-002): for outer fold f the 39-column joint-ridge
procedure (base choice, alpha by inner leave-one-fold-out, fit, stretch) is rerun with fold f's labels hidden, once per column set;
the incumbent t1 (39 columns) is recomputed in the same run (L6).
usage: strict_ladder.py FOLD OUTDIR | strict_ladder.py assemble OUTDIR"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
V = {'t1': [], 't2': ['mtcnn5'], 't2_no_attnp5': ['mtcnn5', 'attnp5'], 't3': ['mtcnn5', 'attnp5', 'cvw3_mean'],
     't3_no_fall16': ['mtcnn5', 'attnp5', 'cvw3_mean', 'ridge_fall16'], 't3x': ['mtcnn5', 'attnp5', 'cvw3_mean', 'ridge_fall16', 'gnn46']}
if sys.argv[1] == 'assemble':
    import chain as C
    y, fold, ok = C.y[:, 2], C.fold, C.ok; Z = [np.load(f'{sys.argv[2]}/lad_f{f}.npz') for f in range(5)]; out = {}
    for v in V:
        P = C.final({f: Z[f][v] for f in range(5)})[:, 2]; E = np.abs(P - y) / np.where(ok, y, 1)
        out[v] = dict(dropped=V[v], all5=round(float(E[ok].mean() * 100), 5), f04=round(float(E[ok & np.isin(fold, [0, 4])].mean() * 100), 5), folds=[round(float(E[ok & (fold == f)].mean() * 100), 4) for f in range(5)])
        out[v]['vs_t1'] = round(out[v]['all5'] - out.get('t1', out[v])['all5'], 5)
    print(json.dumps(out, indent=1)); json.dump(out, open(f'{sys.argv[2]}/strict_ladder.json', 'w'), indent=1); sys.exit(0)
F_OUT, OUT = int(sys.argv[1]), sys.argv[2]
import hexlib as L
L.y[L.fold == F_OUT, :] = 0.0; L.good[L.fold == F_OUT] = False
import chain as C
import joint_fit as JF
JF.ok, JF.good, JF.y = C.ok, C.good, C.y[:, 2]
G = JF.columns()
bases = {'hb': {f: C.L.HB[f].copy() for f in range(5)}, 'ap1_nested': C.STEPS['ap1']({f: C.L.HB[f].copy() for f in range(5)})[0]}
res = {}
for v, drop in V.items():
    names = [k for k in G if k != 'gnn4647' and k not in drop]; X = np.concatenate([G[k] for k in names], 1); outs = {}
    for bn, b in bases.items(): JF.INNER.clear(); r, cd = JF.run(b, X); outs[bn] = (cd, list(JF.INNER.values())[0][F_OUT])
    res[v] = outs[min(outs, key=lambda k: outs[k][1])][0][F_OUT]
np.savez_compressed(f'{OUT}/lad_f{F_OUT}.npz', **res); print('done', F_OUT, flush=True)
