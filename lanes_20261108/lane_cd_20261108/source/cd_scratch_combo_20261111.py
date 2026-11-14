"""From-scratch CD combination search (PI via coordinator, 11-11), CPU only, no training. Pool = every CD member with five-fold OOF predictions.
Fully nested: for each outer fold f, selection, weights, mask-slot choice and stretch use only the other four folds; fold f is scored once.
 M1 forward selection from empty with the usual 0.05 weight grid: each step tries every unused member at weight 0.05..0.50 (the chosen set is
    rescaled by 1-w), keeps the best if it lowers the four-fold objective, then refines weights by moving 0.05 between pairs until no move helps.
 M2 greedy ensemble selection with replacement (Caruana): 40 steps, each adds one copy of the member that lowers the objective most; the step
    with the lowest four-fold objective is kept (weights = copy counts / steps).
 Objective (both): four-fold CD MAPE after the usual stretch (fit_stretch1 on those folds). Combination = log blend.
 Mask slot (after either): variant in {seg_loose cd_mm2 with n_int>=50 (current), cd_mm2 with n_int>=20, detector density s48_top3_tri_med}
    calibrated by a linear fit in log space on the four folds, weight 0..0.2 step 0.05, then the stretch is refit (as in the freeze stack).
Writes reads/scratch_combo_20261111.json and .npz. Workers: 4 processes, one thread each."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ['A2SNAP'] = '1'
import json, multiprocessing as mp, numpy as np
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *


def vec_from(path):
    z = np.load(path); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; return p


def per_fold(fn):
    out = np.full(len(yc), np.nan)
    for f in range(5):
        p = fn(f); out[fold == f] = p[fold == f]
    return out


def attn(track):
    def fn(f):
        r = X / f'lane_backbone_20261108/runs/attnpool_f{f}_dinov3b16_s46/{track}/runs/fold{f}/dinov3_b16'
        L = [np.log(vec_from(r / f'predictions_epoch_{e:02d}.npz')) for e in range(8, 12)]
        return np.exp(0.5 * np.mean(L, 0) + 0.5 * np.log(vec_from(r / 'predictions_final_tta.npz')))
    return per_fold(fn)


def cv_order(seed):
    def fn(f):
        r = X / f'lane_cv_20261108/arms/dinov3_order_f{f}_s{seed}/runs/fold{f}/dinov3_b16'
        return np.exp(np.mean([np.log(vec_from(r / f'predictions_epoch_{e:02d}.npz')) for e in range(16, 21)], 0))
    return per_fold(fn)


POOL = {'P1 (five-member mean, calibrated)': R.P1}
for k, v in R.hist.items():
    POOL[f'P1 member {k}'] = np.asarray(v)[:, 0].astype(float)
for s in (46, 47):
    POOL[f'recipe A s{s} snapshot (TTA e20 + single e8-19)'] = np.exp(0.5 * np.log(R.seeds[s][:, 0]) + 0.5 * np.log(R._rec(s))) * R.sc
    POOL[f'recipe A s{s} final (TTA e20)'] = R.seeds[s][:, 0] * R.sc
for key, name in [('dinov3_s46_tta', 'DINOv3-B s46 TTA e16-20'), ('dinov3_s47_tta', 'DINOv3-B s47 TTA e16-20'), ('convnext_s46_tta', 'ConvNeXt-B s46 TTA e16-20'),
                  ('dinov3_dist_s46_tta', 'DINOv3-B HL-Gauss head s46 TTA e16-20'), ('vitl_s46_tt', 'ViT-L s46 (e8-11 + final TTA)'), ('vitl_s47_tt', 'ViT-L s47 (e8-11 + final TTA)'),
                  ('win2x_s46_tt', 'recipe A 2x-window s46 (backbone lane)')]:
    POOL[name] = R.member(key)
POOL['attention-pool DINOv3-B raw (backbone lane)'] = attn('raw')
POOL['attention-pool DINOv3-B ema998 (backbone lane)'] = attn('ema998')
for s in (46, 47, 49):
    POOL[f'order-loss DINOv3-B s{s} (CV lane, e16-20)'] = cv_order(s)
NAMES = list(POOL)
for n in NAMES:
    assert np.isfinite(POOL[n]).all() and (POOL[n] > 0).all(), n
LOG = np.stack([np.log(POOL[n]) for n in NAMES], 1); M = len(NAMES)
MASKS = {'cd_mm2 n_int>=50': (K.have, K.lm),
         'cd_mm2 n_int>=20': (K.F[:, K.nm['n_int']] >= 20, np.log(np.where(K.F[:, K.nm['n_int']] >= 20, K.F[:, K.nm['cd_mm2']], 1.))),
         'detector density s48_top3_tri_med': (np.isfinite(R.Dr), np.where(np.isfinite(R.Dr), R.Dr, 0.))}  # stored on a log-like scale (no positive values): used linearly


def objective(sel, w, tr):
    q = np.exp(LOG[:, sel] @ w); prm = fit_stretch1(q, tr & good); return cd(apply1(q, prm), tr)


def forward(tr, maxk=8):
    sel, w, best = [], np.zeros(0), np.inf
    while len(sel) < maxk:
        cand = None
        for m in range(M):
            if m in sel: continue
            for a in ([1.0] if not sel else np.round(np.arange(0.05, 0.501, 0.05), 2)):
                ww = np.append(w * (1 - a), a); e = objective(sel + [m], ww, tr)
                if cand is None or e < cand[0]: cand = (e, m, ww)
        if cand[0] >= best - 1e-9: break
        best, sel, w = cand[0], sel + [cand[1]], cand[2]
        for _ in range(4):  # pairwise refinement on the 0.05 grid
            moved = False
            for i in range(len(sel)):
                for j in range(len(sel)):
                    if i == j or w[i] < 0.05 - 1e-9: continue
                    ww = w.copy(); ww[i] -= 0.05; ww[j] += 0.05; ww = np.round(ww, 2); e = objective(sel, ww, tr)
                    if e < best - 1e-9: best, w, moved = e, ww, True
            if not moved: break
        keep = w > 1e-9; sel = [s for s, k in zip(sel, keep) if k]; w = w[keep]
    return sel, w, best


def caruana(tr, steps=40):
    counts = np.zeros(M); hist = []
    for t in range(1, steps + 1):
        cand = None
        for m in range(M):
            c = counts.copy(); c[m] += 1; sel = np.flatnonzero(c); e = objective(list(sel), c[sel] / t, tr)
            if cand is None or e < cand[0]: cand = (e, m)
        counts[cand[1]] += 1; hist.append((cand[0], counts.copy()))
    e, c = min(hist, key=lambda h: h[0]); sel = np.flatnonzero(c); return list(sel), c[sel] / c.sum(), e


def mask_slot(q, tr):
    best = None
    for mn, (have, lm) in MASKS.items():
        trm = tr & have & (yc > 0) & good
        if trm.sum() < 200: continue
        co = np.polyfit(lm[trm], np.log(yc[trm]), 1); Mc = np.polyval(co, lm)
        for wmk in (0, 0.05, 0.1, 0.15, 0.2):
            qq = np.where(have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if best is None or e < best[0]: best = (e, mn if wmk > 0 else 'none', wmk, qq, p2)
    return best


def outer(f):
    tr = fold != f; res = {}
    for meth, fn in (('M1 forward', forward), ('M2 ensemble selection', caruana)):
        sel, w, e_in = fn(tr)
        q = np.exp(LOG[:, sel] @ w); e2, mn, wmk, qq, p2 = mask_slot(q, tr)
        res[meth] = dict(members={NAMES[s]: round(float(x), 3) for s, x in zip(sel, w)}, mask=mn, mask_w=wmk, inner_cd=round(e2, 4),
                         pred=apply1(qq, p2)[fold == f].tolist())
    return f, res


if __name__ == '__main__':
    rows = np.isin(fold, range(5))
    print('pool', M, 'members:', json.dumps(NAMES), flush=True)
    for n in NAMES:
        print(f'  {cd(POOL[n], rows):.3f}  {n}', flush=True)
    with mp.get_context('fork').Pool(4) as pool:
        outs = dict(pool.map(outer, range(5)))
    v3 = np.load(X / 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz'); assert (v3['idx'] == idx).all(); base = v3['oof_cd']
    summary = {'pool': NAMES, 'v3_candidate': round(cd(base, rows), 4)}
    for meth in ('M1 forward', 'M2 ensemble selection'):
        oof = np.full(len(yc), np.nan)
        for f in range(5):
            oof[fold == f] = outs[f][meth]['pred']
        rep(f'{meth} from scratch (nested)', oof, REF, rows=rows)
        rep(f'{meth} vs v3 candidate 6.1741', oof, base, rows=rows)
        summary[meth] = dict(cd=round(cd(oof, rows), 4), delta_vs_v3=round(cd(oof, rows) - cd(base, rows), 4),
                             folds={f: round(cd(oof, fold == f) - cd(base, fold == f), 3) for f in range(5)},
                             picks={f: {k: v for k, v in outs[f][meth].items() if k != 'pred'} for f in range(5)})
        np.save(X / f'lane_cd_20261108/reads/scratch_combo_{meth.split()[0]}_oof_20261111.npy', oof)
        for f in range(5):
            print(meth, 'fold', f, json.dumps({k: v for k, v in outs[f][meth].items() if k != 'pred'}), flush=True)
    json.dump(summary, open(X / 'lane_cd_20261108/reads/scratch_combo_20261111.json', 'w'), indent=1)
    print(json.dumps({k: summary[k]['cd'] if isinstance(summary[k], dict) else summary[k] for k in ('v3_candidate', 'M1 forward', 'M2 ensemble selection')}), flush=True)
