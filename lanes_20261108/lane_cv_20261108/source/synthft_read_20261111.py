"""CV lane (2026-11-11, coordinator for the backbone lane): CV read of the synthetic-pretrained then fine-tuned DINOv3-B
(lane_backbone_20261108/runs/synthft3_f{f}_dinov3b16_s46, raw track, log mean of epochs 8-12, minus log A as every DINOv3 member)
on the folds that exist, strictly nested within those folds.
Procedure (the strict cv_search_20261111 procedure restricted to a fold subset): for outer fold f, weights are chosen by coordinate descent
(grid 0..1 step 0.125, two passes) on the other available folds only, each inner fold g scored with a stretch fit on the available folds
other than f (with two folds that is g itself: nothing from fold f is used); the outer fold is then scored with the stretch fit on the
available folds other than f. The same procedure is applied to (i) the six 10.1101 components (same-folds strict value), (ii) the six plus
the synthetic member (7th component), (iii) the six with the synthetic member in place of the plain DINOv3 term.
usage: synthft_read_20261111.py FOLDS (e.g. 04 or 01234) [PREFIX=synthft3]"""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
import ceiling_meta_cpu_20261103 as C
from pathlib import Path
X = Path('/workspace/clear-ec-research/experiments'); LANE = X / 'lane_cv_20261108'
FOLDS = tuple(int(c) for c in sys.argv[1]); PREFIX = sys.argv[2] if len(sys.argv) > 2 else 'synthft3'
y = BP.y[:, 1]; fold = BP.fold; good = BP.good; ok = y > 0; logH = np.log(BP.H[:, 1]); A = BP._b['A'][:, 1]; GRIDW = np.round(np.arange(0.125, 1.0001, 0.125), 3)
pos = {int(i): n for n, i in enumerate(BP.idx)}; U = np.isin(fold, FOLDS)
syn = np.full(len(y), np.nan)
for f in FOLDS:
    for tgt, root in ((syn, X / f'lane_backbone_20261108/runs/{PREFIX}_f{f}_dinov3b16_s46/raw/runs/fold{f}/dinov3_b16'),):
        out = []
        for e in range(8, 13):
            with np.load(root / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        assert all((o[0] == out[0][0]).all() for o in out); r = np.array([pos[int(i)] for i in out[0][0]])
        assert (fold[r] == f).all(), 'prediction rows are not fold f'; tgt[r] = np.mean([np.log(q[:, 1]) for _, q in out], 0)
assert np.isfinite(syn[U]).all(), 'synthetic member missing rows on the requested folds'
comp_syn = np.where(U, syn - np.log(A), 0.0)
pool = np.load(LANE / 'out/pool_20261111.npz', allow_pickle=True); names = [str(n) for n in pool['names']]
six = [k for k, n in enumerate(names) if n in ('dino', 'fz:spectrum', 'fz:cov_sqrt', 'fz:autocorr') or n.startswith('graph:tta_s46_tta4_5f+') or n.startswith('graph:gord_s46_tta4_5f+')]
assert len(six) == 6; NAMES6 = [names[k] for k in six]; idino = NAMES6.index('dino')
C6 = np.stack([np.stack([pool[f'c{k}_f{f}'] for f in range(5)]) for k in six]); CS = np.stack([comp_syn] * 5)[None]
CONF = {'six (same-folds strict)': (C6, NAMES6), 'six + synth (7th)': (np.concatenate([C6, CS]), NAMES6 + ['synth']),
        'synth in place of dino': (np.concatenate([np.delete(C6, idino, 0), CS]), [n for n in NAMES6 if n != 'dino'] + ['synth'])}
LB = np.stack([logH + BP.BEST[f][:, 1] for f in range(5)])
ape = lambda p, r: np.abs(p[r] - y[r]) / y[r] * 100


def stretch_fit(lp, rows):
    r = rows & ok; lb = lp[r]; mu = lb.mean(); yy = y[r]; best = None
    for g in N.GRID:
        q = np.exp(g * (lb - mu) + mu); c = C.wmed(yy / q, q / yy); m = np.mean(np.abs(c * q - yy) / yy)
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]


def apply_st(lp, prm): g, c, mu = prm; return c * np.exp(g * (lp - mu) + mu)


def run(COMP):
    K = len(COMP)
    def pre(gm, w): return LB[gm] + np.tensordot(w, COMP[:, gm, :], axes=1)
    def inner(w, f):
        tot = 0.0; n = 0
        for g in FOLDS:
            if g == f: continue
            rest = [h for h in FOLDS if h not in (f, g)] or [g]                  # stretch rows: available folds other than f and g; with two folds, g
            lp = pre(g, w); prm = stretch_fit(lp, np.isin(fold, rest) & good); sel = (fold == g) & ok; p = apply_st(lp[sel], prm); tot += np.sum(np.abs(p - y[sel]) / y[sel]); n += sel.sum()
        return tot / n * 100
    def coord(w, obj, passes=2):
        cur = obj(w)
        for _ in range(passes):
            for k in range(K):
                best = (cur, w[k])
                for gv in np.r_[0.0, GRIDW]:
                    if gv == w[k]: continue
                    t = w.copy(); t[k] = gv; s = obj(t)
                    if s < best[0] - 1e-9: best = (s, gv)
                w[k] = best[1]; cur = best[0]
        return w
    P = np.full(len(y), np.nan); W = {}
    for f in FOLDS:
        w = coord(np.zeros(K), lambda w, f=f: inner(w, f)); lp = pre(f, w); prm = stretch_fit(lp, np.isin(fold, [h for h in FOLDS if h != f]) & good)
        P[fold == f] = apply_st(lp[fold == f], prm); W[f] = w
    return P, W


res = {'folds': list(FOLDS), 'prefix': PREFIX}; base = None
for nm, (COMP, cn) in CONF.items():
    P, W = run(COMP); m = float(ape(P, U & ok).mean()); pf = {f: float(ape(P, (fold == f) & ok).mean()) for f in FOLDS}
    if base is None: base = (m, pf)
    res[nm] = dict(cv=round(m, 4), delta=round(m - base[0], 4), per_fold_delta={f: round(pf[f] - base[1][f], 4) for f in FOLDS},
                   weights={f: {cn[k]: float(W[f][k]) for k in range(len(cn)) if W[f][k] > 0} for f in FOLDS})
    print(json.dumps({nm: res[nm]}), flush=True)
r = U & ok
res['member_alone_cv_mape'] = round(float(np.mean(np.abs(np.exp(syn[r]) - y[r]) / y[r]) * 100), 3)
json.dump(res, open(LANE / f'out/synthft_read_{"".join(map(str, FOLDS))}_20261111.json', 'w'), indent=1); print(json.dumps({k: res[k] for k in ('member_alone_cv_mape',)}), flush=True)
