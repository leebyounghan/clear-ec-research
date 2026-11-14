"""Repeated grouped 10-fold read at the stack layer (PI via coordinator 00:30 EST 11-14). Rows = rows with an out-of-fold prediction of the arm
(ROWS env: 'fold0' or 'all'). Groups = donor key (BP.donors = slide_id), with old-format serial-gap-1 pairs merged (union-find). 20 repeats x 10
groups (seeded). In each split every stack constant the strict protocol fits (weights on a coarse grid, mask slot, stretch) is fitted on 9 groups
and scored on the held-out one; member scalars (P1, recipe A) stay as in the strict protocol (fitted on the folds other than the row's fold).
Variants (VARS env): inc, armD (arm in D's s46 half), armX (arm as an extra member), pD, pX (control P in the same places).
Members: default arm and P as exp(mean log single view e8-12) on fold 0; ARMFILE / PFILE env = npz files (e16-20 TTA) instead. mode run -> reads/cv10/<tag>_<var>.npz; mode agg -> reads/cv10_<tag>.json."""
import os, sys, re, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
MODE = sys.argv[1]; TAG = os.environ.get('TAG', 'elr2e5_f0_sv812'); OUT = LANE / 'reads/cv10'; OUT.mkdir(exist_ok=True)
ROWSEL = os.environ.get('ROWS', 'fold0'); rows = (fold == 0) if ROWSEL == 'fold0' else np.ones(len(yc), bool)
rows &= yc > 0; idxr = np.where(rows)[0]; NR, NG = 20, 10
# groups: donor key, old-format gap-1 serial pairs merged
ids_s = np.array([str(i).strip() for i in ids]); par = {}
def find(a):
    while par.get(a, a) != a: a = par[a]
    return a
def union(a, b): par[find(a)] = find(b)
dk = np.array([str(d) for d in donors]); old = np.array([not re.match(r'^\d{4}-\d{4}', i) for i in ids_s])
ser = {}
for i in np.where(old)[0]:
    s, y_ = ids_s[i].split('-')[0], ids_s[i].split('-')[1][:2]; ser[(y_, int(s))] = i
for (y_, s), i in ser.items():
    j = ser.get((y_, s + 1))
    if j is not None: union(dk[i], dk[j])
grp = np.array([find(d) for d in dk])
def vec(p):
    z = np.load(p); v = np.full(len(yc), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; return v
def sv(root): d = LANE / 'arms' / root / 'raw/runs/fold0/dinov3_b16'; return np.exp(np.mean([np.log(vec(d / f'predictions_epoch_{e:02d}.npz')) for e in range(8, 13)], 0))
if os.environ.get('ARMFILE'):   # MEMBER = e16-20 four-flip TTA (or a final-TTA file): arm and control P read from npz files under the lane folder
    ARM = vec(LANE / os.environ['ARMFILE']); PC = vec(LANE / os.environ.get('PFILE', 'tta/screen_plain_f0_tta4_e16_20.npz'))
else:
    ARM = sv(os.environ.get('ARMROOT', 'sweep_elr2e5_f0_s46')); PC = sv('screen_plain_f0_s46')
SCALE_CD = 0.9970446110788349; P1raw = R.P1 / SCALE_CD; Afin_raw = 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0])
D46 = R.member('dinov3_s46_tta'); D47 = R.member('dinov3_s47_tta'); Dm = np.exp(0.5 * np.log(D46) + 0.5 * np.log(D47))
def vit(sd):
    o = np.full(len(yc), np.nan)
    for f in range(5):
        r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
        o[fold == f] = vec(r / 'predictions_final_tta.npz')[fold == f]
    return o
Vf = np.exp(0.5 * np.log(vit('s46')) + 0.5 * np.log(vit('s47')))
def wsc(p, tr): k = tr & (yc > 0); return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))
sP = np.zeros(len(yc)); sA = np.zeros(len(yc))
for f in range(5):
    m = fold == f; sP[m] = wsc(P1raw, fold != f); sA[m] = wsc(Afin_raw, fold != f)
gm = lambda a, b: np.exp(0.5 * np.log(a) + 0.5 * np.log(b))
VAR = {'inc': ([Dm], None), 'armD': ([gm(ARM, D47)], None), 'armX': ([Dm], ARM), 'pD': ([gm(PC, D47)], None), 'pX': ([Dm], PC)}
GRID = N.GRID
def stretch_fit(lq, y):
    mu = lq.mean(); best = None
    for g in GRID:
        q = np.exp(g * (lq - mu) + mu); c = C.wmed(y / q, q / y); m = np.mean(np.abs(c * q - y) / y)
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]
def apply_st(lq, p): g, c, mu = p; return c * np.exp(g * (lq - mu) + mu)
def fit_apply(tr, te, Dk, extra):
    L = [np.log(P1raw * sP), np.log(Afin_raw * sA), np.log(Dk), np.log(Vf)] + ([np.log(extra)] if extra is not None else [])
    L = np.stack(L, 1); y = yc; grids = []
    for a in (0.3, 0.4, 0.5, 0.6, 0.7):
        for mw in itertools.product((0, 0.1, 0.2, 0.3, 0.4), repeat=2):
            for xw in ((0, 0.1, 0.2, 0.3) if extra is not None else (None,)):
                r_ = 1 - a - sum(mw) - (xw or 0)
                if r_ >= 0.099: grids.append([r_, a, *mw] + ([xw] if xw is not None else []))
    best = None; trg = tr & good
    for w in grids:
        lq = L[trg] @ np.array(w); st = stretch_fit(lq, y[trg]); e = np.mean(np.abs(apply_st(L[tr] @ np.array(w), st) - y[tr]) / y[tr])
        if best is None or e < best[0]: best = (e, np.array(w))
    w = best[1]; lq_all = L @ w; trm = trg & K.have; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
    for wmk in (0, 0.05, 0.1, 0.15, 0.2):
        qq = np.where(K.have, (1 - wmk) * lq_all + wmk * Mc, lq_all); st = stretch_fit(qq[trg], y[trg]); e = np.mean(np.abs(apply_st(qq[tr], st) - y[tr]) / y[tr])
        if b2 is None or e < b2[0]: b2 = (e, wmk, st)
    qq = np.where(K.have, (1 - b2[1]) * lq_all + b2[1] * Mc, lq_all); return apply_st(qq[te], b2[2])
if MODE == 'run':
    ug = np.unique(grp[idxr])
    for v in os.environ['VARS'].split(','):
        Dk, extra = VAR[v][0][0], VAR[v][1]; P = np.full((NR, len(yc)), np.nan)
        for r in range(NR):
            rng = np.random.default_rng(1000 + r); lab = dict(zip(ug, rng.permutation(len(ug)) % NG)); gi = np.array([lab[g] for g in grp[idxr]])
            for k in range(NG):
                te = np.zeros(len(yc), bool); te[idxr[gi == k]] = True; tr = rows & ~te
                P[r, te] = fit_apply(tr, te, Dk, extra)
            print(v, 'repeat', r, round(float(np.mean(np.abs(P[r, idxr] - yc[idxr]) / yc[idxr]) * 100), 4), flush=True)
        np.savez_compressed(OUT / f'{TAG}_{v}.npz', P=P, rows=rows)
else:
    Z = {v: np.load(OUT / f'{TAG}_{v}.npz')['P'] for v in ('inc', 'armD', 'armX', 'pD', 'pX') if (OUT / f'{TAG}_{v}.npz').exists()}
    ape = {v: np.abs(P[:, idxr] - yc[idxr]) / yc[idxr] * 100 for v, P in Z.items()}
    _t1 = np.load(LANE / 'reads/cd_tier_t1_20261112.npz')['oof_cd']; ex = (np.abs(_t1 / np.where(yc > 0, yc, 1) - 1) > 1.0)[idxr]
    res = {'tag': TAG, 'rows': int(len(idxr)), 'groups': int(len(np.unique(grp[idxr]))), 'repeats': NR, 'per_variant_mean_cd': {v: round(float(a.mean(1).mean()), 4) for v, a in ape.items()}}
    g = grp[idxr]; u, inv = np.unique(g, return_inverse=True); rng = np.random.default_rng(0); B = [rng.integers(0, len(u), len(u)) for _ in range(2000)]
    def comp(a, b, mask=None):
        mask = np.ones(len(idxr), bool) if mask is None else mask
        per_rep = (a[:, mask].mean(1) - b[:, mask].mean(1)); d = (a.mean(0) - b.mean(0)) * mask
        s = np.bincount(inv, d, len(u)); n = np.bincount(inv, mask.astype(float), len(u)); bs = np.array([s[k].sum() / max(n[k].sum(), 1) for k in B])
        return dict(mean_diff=round(float(per_rep.mean()), 4), sd_across_repeats=round(float(per_rep.std()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], share_lower=round(float(np.mean(bs < 0)), 3))
    for v in ape:
        if v != 'inc': res[f'{v}_vs_inc'] = comp(ape[v], ape['inc'])
    if 'armD' in ape and 'pD' in ape: res['armD_vs_pD'] = comp(ape['armD'], ape['pD'])
    if 'armX' in ape and 'pX' in ape: res['armX_vs_pX'] = comp(ape['armX'], ape['pX'])
    if ex.any():
        res['ex100'] = {f'{v}_vs_inc': comp(ape[v], ape['inc'], ~ex) for v in ape if v != 'inc'}
    print(json.dumps(res, indent=1)); json.dump(res, open(LANE / f'reads/cv10_{TAG}.json', 'w'), indent=1)
