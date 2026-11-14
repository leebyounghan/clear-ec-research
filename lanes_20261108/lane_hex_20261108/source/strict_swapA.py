"""D-SYN-003 HEX read (HEX lane, 2026-11-11): recipe A with the synthetic-pretrained s46 in place of plain s46 on folds 0 and 4
(the CD lane's file, Q-HEX-001 / A-CD-SWAPA-PLAN) entered everywhere the HEX stack uses recipe A: A itself (blend columns, ap1 base),
B = 0.5 (A + P1) and H (best-pipeline base, h6 blend). Strict: for outer fold f the 39-column joint-ridge procedure is rerun with
fold f's labels hidden; the incumbent (plain A) is recomputed in the same run (L6).
usage: strict_swapA.py FOLD OUTDIR SWAPPED.npz | strict_swapA.py assemble OUTDIR"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')

if sys.argv[1] == 'assemble':
    import chain as C
    y, fold, ok = C.y[:, 2], C.fold, C.ok; Z = [np.load(f'{sys.argv[2]}/swa_f{f}.npz') for f in range(5)]; out = {}
    for v in ('plainA', 'swapA'):
        P = C.final({f: Z[f][v] for f in range(5)})[:, 2]; E = np.abs(P - y) / np.where(ok, y, 1)
        out[v] = dict(all5=round(float(E[ok].mean() * 100), 5), f04=round(float(E[ok & np.isin(fold, [0, 4])].mean() * 100), 5), folds=[round(float(E[ok & (fold == f)].mean() * 100), 4) for f in range(5)])
    out['delta'] = dict(all5=round(out['swapA']['all5'] - out['plainA']['all5'], 5), f04=round(out['swapA']['f04'] - out['plainA']['f04'], 5), folds=[round(a - b, 4) for a, b in zip(out['swapA']['folds'], out['plainA']['folds'])])
    print(json.dumps(out, indent=1)); json.dump(out, open(f'{sys.argv[2]}/strict_swapA.json', 'w'), indent=1); sys.exit(0)

F_OUT, OUT, SW = int(sys.argv[1]), sys.argv[2], sys.argv[3]
import hexlib as L
L.y[L.fold == F_OUT, :] = 0.0; L.good[L.fold == F_OUT] = False
import chain as C
import joint_fit as JF
JF.ok, JF.good, JF.y = C.ok, C.good, C.y[:, 2]
BP = C.BP; A0, P1, B0, H0 = BP._b['A'].copy(), BP._b['P1'].copy(), BP._b['B'].copy(), BP.H.copy()
z = np.load(SW); assert (z['indices'] == C.idx).all(); assert np.allclose(z['A_plain'], A0, rtol=0, atol=1e-9)


def set_A(A):
    BP._b['A'][:] = A; B = 0.5 * (A + P1); BP._b['B'][:] = B; H = B.copy(); h = BP._have
    H[h, 2] = np.exp(0.72 * np.log(B[h, 2]) + 0.28 * (-0.2084 + 0.5989 * BP.f6[h])); BP.H[:] = H
    for f in range(5): C.L.HB[f] = H[:, 2] * np.exp(BP.BEST[f][:, 2])


res = {}
for v, A in (('plainA', A0), ('swapA', z['A'].astype(float))):
    set_A(A); G = JF.columns(); names = [k for k in G if k != 'gnn4647']; X = np.concatenate([G[k] for k in names], 1)
    bases = {'hb': {f: C.L.HB[f].copy() for f in range(5)}, 'ap1_nested': C.STEPS['ap1']({f: C.L.HB[f].copy() for f in range(5)})[0]}; outs = {}
    for bn, b in bases.items(): JF.INNER.clear(); r, cd = JF.run(b, X); outs[bn] = (cd, list(JF.INNER.values())[0][F_OUT])
    res[v] = outs[min(outs, key=lambda k: outs[k][1])][0][F_OUT]
set_A(A0)
np.savez_compressed(f'{OUT}/swa_f{F_OUT}.npz', **res); print('done', F_OUT, flush=True)
