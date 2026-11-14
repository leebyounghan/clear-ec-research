"""Strictly nested HEX values (HEX lane, 2026-11-11, coordinator): for outer fold f the whole procedure is rerun with fold f's labels
hidden (HEX label set to 0 -> not ok, good False), so no constant (option pick, ridge alpha or weights, share, stretch, keep
decision, base choice) can see fold f, not even through the other folds' candidates. The fold-f candidate from that run is kept;
assemble.py then scores each row with its own fold's candidate and a stretch fit on the other folds.
usage: strict_nest.py FOLD OUTDIR"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
sys.path.insert(0, LANE + '/source')
F_OUT = int(sys.argv[1]); OUT = sys.argv[2]
import hexlib as L
L.y[L.fold == F_OUT, :] = 0.0; L.good[L.fold == F_OUT] = False          # hide fold F_OUT before chain computes ok
import chain as C
assert not C.ok[C.fold == F_OUT].any() and not C.good[C.fold == F_OUT].any()
import combo_search as CS
CS.ok = C.ok; CS.y = C.y[:, 2]
import joint_fit as JF
JF.ok, JF.good, JF.y = C.ok, C.good, C.y[:, 2]
STEPS10 = 'ap1 h6w graphens imgstats prior ridgeev ridgeev_fall16 segm cvw3m_fix mtcnn5_fix'.split()
res = {}
# (a) the fixed 10-step chain, every step applied (its composition was fixed on all folds, so only the fits are strict here)
cand = {f: C.L.HB[f].copy() for f in range(5)}
for s in STEPS10: cand = CS.apply(s, cand)[0]
res['chain10_fixed'] = cand[F_OUT]
# (b) the chain procedure: historical order, a step kept for a fold only if it lowers HEX on that fold's other (visible) folds
cand = {f: C.L.HB[f].copy() for f in range(5)}; cur, _, _ = CS.scores(cand); kept = []
for s in STEPS10:
    new = CS.apply(s, cand)[0]; i1, _, _ = CS.scores(new); nc = dict(cand)
    for f in range(5):
        if i1[f] < cur[f]: nc[f] = new[f]; cur[f] = i1[f]; kept.append((f, s)) if f == F_OUT else None
    cand = nc
res['chain_procedure'] = cand[F_OUT]
# (c) joint ridge, deployable columns, nested base choice, both base conventions
G = JF.columns(); names = [k for k in G if k not in ('attnp5', 'gnn4647')]; X = np.concatenate([G[k] for k in names], 1)
names2 = [k for k in G if k not in ('gnn4647',)]; X2 = np.concatenate([G[k] for k in names2], 1)
bases = {'hb': {f: C.L.HB[f].copy() for f in range(5)}, 'ap1_nested': C.STEPS['ap1']({f: C.L.HB[f].copy() for f in range(5)})[0]}
for tag, XX in (('joint', X), ('joint_attnp', X2)):
    outs = {}
    for bn, b in bases.items(): JF.INNER.clear(); r, cd = JF.run(b, XX); outs[bn] = (cd, dict(JF.INNER))
    inner = {bn: list(v[1].values())[0][F_OUT] for bn, v in outs.items()}; pick = min(inner, key=inner.get)
    res[tag] = outs[pick][0][F_OUT]; res[tag + '_base'] = pick
np.savez_compressed(f'{OUT}/strict_f{F_OUT}.npz', **{k: v for k, v in res.items() if not k.endswith('_base')})
json.dump(dict(fold=F_OUT, kept_for_this_fold=[s for _, s in kept], bases={k: v for k, v in res.items() if k.endswith('_base')}), open(f'{OUT}/strict_f{F_OUT}.json', 'w'))
print('done fold', F_OUT, flush=True)
