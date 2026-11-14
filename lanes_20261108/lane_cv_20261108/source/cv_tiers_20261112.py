"""CV lane (2026-11-12, D-PKG-002): reduced deploy tiers of the CV stack, strict five folds, incumbent recomputed in the same code.
Procedure = cv_search_20261111 'fixed' (strict): for outer fold f, coordinate descent (grid 0..1 step 0.125, two passes) on the other four folds,
inner fold g scored with a stretch fit on folds not in {f, g}; outer fold scored with the stretch fit on folds != f. Deploy weights = the same
coordinate descent with no outer fold (score_all: fold g stretch on folds != g), i.e. the nested procedure with no outer fold (L5).
DINOv3 term variants (CV column, log CV minus log A, single view, as the freeze's plain DINOv3 term):
  d3   seeds 46/47/49, mean log over epochs 16-20 (the D-CV-001 term; checked equal to the pool's 'dino')
  d1   seed 46 only (deploy model s46b, shared with the CD lane's D), epochs 16-20
  d2   seeds 46 + 47 (both in the CD lane's D), epochs 16-20
  d1e  seed 46 only, epoch 20 only (one checkpoint)
Configs (name: components): incumbent: d3 + fz x3 + quartets x2 | t1: d1 + fz x3 + quartets x2 | two_seed: d2 + ... | t3_e20: d1e + ... |
  t3_nodino: fz x3 + quartets x2 | t1_noquart: d1 + fz x3 | t3_fzonly: fz x3
usage: cv_tiers_20261112.py CONFIG      writes out/cv_tier_<CONFIG>_20261112.{npz,json} and _comps.npz (stack-file layout for cv_deploy_fit)"""
import json, sys, time
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
import ceiling_meta_cpu_20261103 as C
from pathlib import Path
X = Path('/workspace/clear-ec-research/experiments'); LANE = X / 'lane_cv_20261108'; CONFIG = sys.argv[1]
y = BP.y[:, 1]; fold = BP.fold; good = BP.good; ok = y > 0; logH = np.log(BP.H[:, 1]); A = BP._b['A'][:, 1]; GRIDW = np.round(np.arange(0.125, 1.0001, 0.125), 3)
pos = {int(i): n for n, i in enumerate(BP.idx)}


def root(sd, f):
    if sd == 46 and f in (0, 4): return X / f'lane_gpu3_20261106/runs/dinov3_b16_f{f}_s46'
    if sd == 47 and f in (0, 4): return X / f'lane_gpu2_20261106/runs/dinov3_b16_f{f}_s47'
    return X / ('lane_cd_20261108/arms' if sd in (46, 47) else 'lane_cv_20261108/arms') / f'dinov3_b16_f{f}_s{sd}'


def dino(seeds, epochs):
    v = np.zeros(len(y))
    for f in range(5):
        per = []
        for sd in seeds:
            run = root(sd, f) / 'runs' / f'fold{f}' / 'dinov3_b16'; out = []
            for e in epochs:
                with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
            assert all(np.array_equal(o[0], out[0][0]) for o in out); per.append((out[0][0], np.mean([np.log(p) for _, p in out], 0)[:, 1]))
        r = np.array([pos[int(i)] for i in per[0][0]]); assert all(np.array_equal(per[0][0], q[0]) for q in per) and (fold[r] == f).all()
        v[r] = np.mean([q[1] for q in per], 0) - np.log(A[r])
    return v


pool = np.load(LANE / 'out/pool_20261111.npz', allow_pickle=True); pn = [str(n) for n in pool['names']]
def pc(name): k = pn.index(name); return np.stack([pool[f'c{k}_f{f}'] for f in range(5)])
FZ = ['fz:spectrum', 'fz:cov_sqrt', 'fz:autocorr']; QP = next(n for n in pn if n.startswith('graph:tta_s46_tta4_5f+')); QO = next(n for n in pn if n.startswith('graph:gord_s46_tta4_5f+'))
D = {'d3': ((46, 47, 49), range(16, 21)), 'd1': ((46,), range(16, 21)), 'd2': ((46, 47), range(16, 21)), 'd1e': ((46,), (20,))}
CONF = {'incumbent': 'd3', 't1': 'd1', 'two_seed': 'd2', 't3_e20': 'd1e', 't3_nodino': None, 't1_noquart': 'd1', 't3_fzonly': None}
dv = CONF[CONFIG]; info = {'config': CONFIG}
check = float(np.max(np.abs(dino(*D['d3']) - pool[f'c{pn.index("dino")}_f0']))); info['d3_vs_pool_dino_max_abs'] = check; assert check < 1e-9, check
names, comps = [], []
if dv: names.append('dino'); comps.append(np.stack([dino(*D[dv])] * 5)); info['dino'] = dict(seeds=list(D[dv][0]), epochs=list(D[dv][1]), view='single')
names += FZ; comps += [pc(n) for n in FZ]
if CONFIG not in ('t1_noquart', 't3_fzonly'): names += [QP, QO]; comps += [pc(QP), pc(QO)]
COMP = np.stack(comps); K = len(COMP); LB = np.stack([logH + BP.BEST[f][:, 1] for f in range(5)])


def stretch_fit(lp, rows):
    r = rows & ok; lb = lp[r]; mu = lb.mean(); yy = y[r]; best = None
    for g in N.GRID:
        q = np.exp(g * (lb - mu) + mu); c = C.wmed(yy / q, q / yy); m = np.mean(np.abs(c * q - yy) / yy)
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]


def apply_st(lp, prm): g, c, mu = prm; return c * np.exp(g * (lp - mu) + mu)
def pre(gm, w): return LB[gm] + np.tensordot(w, COMP[:, gm, :], axes=1)


def inner(w, f):
    tot = 0.0; n = 0
    for g in range(5):
        if g == f: continue
        lp = pre(g, w); prm = stretch_fit(lp, (fold != g) & (fold != f) & good); sel = (fold == g) & ok; tot += np.sum(np.abs(apply_st(lp[sel], prm) - y[sel]) / y[sel]); n += sel.sum()
    return tot / n * 100


def score_all(w):
    tot = 0.0; n = 0
    for g in range(5):
        lp = pre(g, w); prm = stretch_fit(lp, (fold != g) & good); sel = (fold == g) & ok; tot += np.sum(np.abs(apply_st(lp[sel], prm) - y[sel]) / y[sel]); n += sel.sum()
    return tot / n * 100


def coord(obj):
    w = np.zeros(K); cur = obj(w)
    for _ in range(2):
        for k in range(K):
            best = (cur, w[k])
            for gv in np.r_[0.0, GRIDW]:
                if gv == w[k]: continue
                t = w.copy(); t[k] = gv; s = obj(t)
                if s < best[0] - 1e-9: best = (s, gv)
            w[k] = best[1]; cur = best[0]
    return w, cur


t0 = time.time(); P = np.full(len(y), np.nan); W = {}
for f in range(5):
    w, _ = coord(lambda w, f=f: inner(w, f)); lp = pre(f, w); prm = stretch_fit(lp, (fold != f) & good); P[fold == f] = apply_st(lp[fold == f], prm); W[f] = w
w_all, obj_all = coord(score_all)
ape = lambda p, r: np.abs(p[r] - y[r]) / y[r] * 100
info.update(names=names, strict_cv=round(float(ape(P, ok).mean()), 4), per_fold={f: round(float(ape(P, ok & (fold == f)).mean()), 4) for f in range(5)},
            weights_per_fold={f: dict(zip(names, map(float, W[f]))) for f in range(5)}, deploy_weights_no_outer_fold=dict(zip(names, map(float, w_all))), all_row_objective=round(obj_all, 4), seconds=round(time.time() - t0))
np.savez_compressed(LANE / f'out/cv_tier_{CONFIG}_20261112.npz', P=np.stack([BP.H[:, 0] * np.nan, P, BP.H[:, 2] * np.nan], 1), names=np.array(names), w_all=w_all, **{f'w{f}': W[f] for f in range(5)})
np.savez_compressed(LANE / f'out/cv_tier_{CONFIG}_20261112_comps.npz', names=np.array(names), **{f'c{k}_f{f}': COMP[k, f] for k in range(K) for f in range(5)})
json.dump(info, open(LANE / f'out/cv_tier_{CONFIG}_20261112.json', 'w'), indent=1); print(json.dumps(info), flush=True)
