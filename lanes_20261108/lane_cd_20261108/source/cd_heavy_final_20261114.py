"""Final heavy-member read (coordinator 03:58 / 06:49 / 07:06 EST 11-14), all rows, CPU, true form. Arms: DINOv3-B s46 encoder LR 2e-5
(e16-20 four-flip TTA; fold 0 = the sweep arm, folds 1-4 = fu2_elr2e5), ViT-L s46 encoder LR 2e-5 (final.pt four-flip TTA, raw track; folds
0 / 4 = sweep_vitl_elr2e5, folds 1-3 = vitlfu_elr2e5), ViT-L s47 encoder LR 2e-5 (vitl47fu_elr2e5, same form). Variants (each only when its
members exist on every row): tierp, p2 (D = s46b alone), dA47 (DINOv3 arm in D's s47 half), vA46 (ViT-L arm in V's s46 half), vA46_dA47,
vA46_p2, vAboth (both V halves replaced), vAboth_dA47, vAboth_p2. MODE strict: strict five folds (machinery of cd_dropladder_20261114.py).
MODE cv10: repeated grouped 10-fold at the stack layer (20 repeats, coarse grid, member scalars strict), donor-clustered interval. Every
variant is compared with tier p, p2 and (when present) p3 = vA46_p2 on all rows, folds 1-4 (fold 0 apart), folds 1-3 (folds 0 + 4 apart), each with and without the
11 rows above 100 % APE in the t1 OOF. SELFTEST=1 substitutes the incumbent members for the arms (every swap must reproduce tierp / p2).
Writes reads/heavy_final_<mode><OUTTAG>_20261114.json."""
import os, sys, re, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
MODE = sys.argv[1]; ok = yc > 0
SCALE_CD = 0.9970446110788349; P1raw = R.P1 / SCALE_CD; Afin_raw = 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0])
D46 = R.member('dinov3_s46_tta'); D47 = R.member('dinov3_s47_tta')
def vec(p):
    z = np.load(p); v = np.full(len(yc), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; return v
def vit(sd):
    o = np.full(len(yc), np.nan)
    for f in range(5):
        r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
        o[fold == f] = vec(r / 'predictions_final_tta.npz')[fold == f]
    return o
V46 = vit('s46'); V47 = vit('s47'); gm = lambda *a: np.exp(np.mean([np.log(x) for x in a], 0))
TAG = os.environ.get('OUTTAG', ''); ST = os.environ.get('SELFTEST') == '1'
def vtta(dirs):
    o = np.full(len(yc), np.nan)
    for f, d in dirs.items():
        p = d / 'predictions_final_tta.npz'
        if p.exists(): o[fold == f] = vec(p)[fold == f]
    return o
VA46 = vtta({f: (LANE / f'arms/sweep_vitl_elr2e5_f{f}_s46' if f in (0, 4) else LANE / f'arms/vitlfu_elr2e5_f{f}_s46') / f'raw/runs/fold{f}/dinov3_l16t' for f in range(5)})
VA47 = vtta({f: LANE / f'arms/vitl47fu_elr2e5_f{f}_s47/raw/runs/fold{f}/dinov3_l16t' for f in range(5)})
DARM = R.member('fu2_elr2e5_tta')
if ST: VA46, VA47, DARM = V46.copy(), V47.copy(), D47.copy()
have = {k: bool(np.isfinite(v[ok]).all()) for k, v in (('DARM', DARM), ('VA46', VA46), ('VA47', VA47))}
Vp = gm(V46, V47); Dp = gm(D46, D47)
CAND = {'tierp': ([Dp, Vp], ()), 'p2': ([D46, Vp], ()), 'dA47': ([gm(D46, DARM), Vp], ('DARM',)),
        'vA46': ([Dp, gm(VA46, V47)], ('VA46',)), 'vA46_dA47': ([gm(D46, DARM), gm(VA46, V47)], ('DARM', 'VA46')), 'vA46_p2': ([D46, gm(VA46, V47)], ('VA46',)),
        'vAboth': ([Dp, gm(VA46, VA47)], ('VA46', 'VA47')), 'vAboth_dA47': ([gm(D46, DARM), gm(VA46, VA47)], ('DARM', 'VA46', 'VA47')), 'vAboth_p2': ([D46, gm(VA46, VA47)], ('VA46', 'VA47')),
        'vA47_p2': ([D46, gm(V46, VA47)], ('VA47',))}   # p3b with the s46 half back at encoder LR 1e-5 (the plain runs), coordinator 14:47
want = os.environ.get('VARIANTS', '')
VARS = {k: (ext, True) for k, (ext, need) in CAND.items() if all(have[n] for n in need) and (not want or k in want.split(',') or k in ('tierp', 'p2'))}
print('members complete:', have, 'variants:', list(VARS), flush=True)
def wsc(p, tr): k = tr & ok; return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))
def stretch_fit(lq, y):
    mu = lq.mean(); best = None
    for g in N.GRID:
        q = np.exp(g * (lq - mu) + mu); c = C.wmed(y / q, q / y); m = np.mean(np.abs(c * q - y) / y)
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]
def apply_st(lq, p): g, c, mu = p; return c * np.exp(g * (lq - mu) + mu)
def fit_apply(tr, te, sP, sA, ext, usemask, coarse):
    L = np.stack([np.log(P1raw * sP), np.log(Afin_raw * sA)] + [np.log(m) for m in ext], 1); grids = []
    Aw = (0.35, 0.45, 0.55) if coarse else np.round(np.arange(0.25, 0.751, 0.05), 3); Ow = (0, 0.1, 0.2, 0.3) if coarse else np.round(np.arange(0, 0.401, 0.05), 3)
    for a in Aw:
        for mw in itertools.product(Ow, repeat=len(ext)):
            r_ = 1 - a - sum(mw)
            if r_ >= 0.099: grids.append(np.array([r_, a, *mw]))
    trg = tr & good & ok; best = None
    for w in grids:
        st = stretch_fit(L[trg] @ w, yc[trg]); e = np.mean(np.abs(apply_st(L[tr & ok] @ w, st) - yc[tr & ok]) / yc[tr & ok])
        if best is None or e < best[0]: best = (e, w)
    w = best[1]; lq = L @ w; trm = trg & K.have; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
    for wmk in ((0, 0.05, 0.1, 0.15, 0.2) if usemask else (0,)):
        qq = np.where(K.have, (1 - wmk) * lq + wmk * Mc, lq); st = stretch_fit(qq[trg], yc[trg]); e = np.mean(np.abs(apply_st(qq[tr & ok], st) - yc[tr & ok]) / yc[tr & ok])
        if b2 is None or e < b2[0]: b2 = (e, wmk, st)
    qq = np.where(K.have, (1 - b2[1]) * lq + b2[1] * Mc, lq); return apply_st(qq[te], b2[2]), (w.round(3).tolist(), b2[1])
sPr = np.zeros(len(yc)); sAr = np.zeros(len(yc))
for f in range(5):
    m = fold == f; sPr[m] = wsc(P1raw, fold != f); sAr[m] = wsc(Afin_raw, fold != f)
_t1 = np.load(LANE / 'reads/cd_tier_t1_20261112.npz')['oof_cd']; EX = ok & (np.abs(_t1 / np.where(ok, yc, 1) - 1) > 1.0)
res = {'mode': MODE, 'selftest': ST, 'members_complete': have, 'ex100_rows': int(EX.sum()), 'member': {}}
SUB = {'all': ok, 'folds1_4': ok & (fold != 0), 'folds1_3': ok & ~np.isin(fold, [0, 4]), 'folds0_4': ok & np.isin(fold, [0, 4])}
for nm_, arm, refs in (('DARM', DARM, (('s46b', D46), ('s47', D47))), ('VA46', VA46, (('s46', V46), ('s47', V47))), ('VA47', VA47, (('s46', V46), ('s47', V47)))):
    if not have[nm_]: continue
    res['member'][nm_] = {'per_fold_cd': {f'fold{f}': round(cd(arm, fold == f), 4) for f in range(5)}}
    for rn, ref in refs:
        res['member'][nm_][f'ref_{rn}_per_fold_cd'] = {f'fold{f}': round(cd(ref, fold == f), 4) for f in range(5)}
        for sn, m in SUB.items():
            d_, ci_ = boot(arm, ref, m); res['member'][nm_][f'vs_{rn}_{sn}'] = dict(diff=round(d_, 4), ci95=[round(x, 4) for x in ci_])
if MODE == 'strict':
    OOF = {}
    for v, (ext, um) in VARS.items():
        o = np.full(len(yc), np.nan)
        for f in range(5):
            tr = fold != f; te = fold == f; p, ww = fit_apply(tr, te, wsc(P1raw, tr), wsc(Afin_raw, tr), ext, um, False); o[te] = p
        OOF[v] = o; print(v, round(cd(o), 4), flush=True)
    for v, o in OOF.items():
        r_ = dict(strict=round(cd(o), 4), strict_ex100=round(cd(o, ~EX), 4))
        for refn in [r_n for r_n in ('tierp', 'p2', 'vA46_p2', 'vAboth_p2') if r_n in OOF]:   # vA46_p2 = p3 (added 10:30 EST 11-14, coordinator 10:36)
            ref = OOF[refn]
            for sn, m in SUB.items():
                d_, ci_ = boot(o, ref, m)
                r_[f'vs_{refn}_{sn}'] = dict(delta=round(cd(o, m) - cd(ref, m), 4), ci95=[round(x, 4) for x in ci_], delta_ex100=round(cd(o, m & ~EX) - cd(ref, m & ~EX), 4))
            r_[f'vs_{refn}_per_fold'] = {f: round(cd(o, fold == f) - cd(ref, fold == f), 4) for f in range(5)}
        res[v] = r_
    np.savez_compressed(LANE / f'reads/heavy_final_strict_oof{TAG}_20261114.npz', **OOF)
else:
    ids_s = np.array([str(i).strip() for i in ids]); par = {}
    def find(a):
        while par.get(a, a) != a: a = par[a]
        return a
    dk = np.array([str(d) for d in donors]); old = np.array([not re.match(r'^\d{4}-\d{4}', i) for i in ids_s]); ser = {}
    for i in np.where(old)[0]: ser[(ids_s[i].split('-')[1][:2], int(ids_s[i].split('-')[0]))] = i
    for (y_, s), i in ser.items():
        j = ser.get((y_, s + 1))
        if j is not None: par[find(dk[i])] = find(dk[j])
    grp = np.array([find(d) for d in dk]); idxr = np.where(ok)[0]; ug = np.unique(grp[idxr]); NR = int(os.environ.get('NR', '20'))
    APE = {}
    for v, (ext, um) in VARS.items():
        A = np.zeros((NR, len(idxr)))
        for r in range(NR):
            rng = np.random.default_rng(2000 + r); lab = dict(zip(ug, rng.permutation(len(ug)) % 10)); gi = np.array([lab[g] for g in grp[idxr]]); P = np.full(len(yc), np.nan)
            for k in range(10):
                te = np.zeros(len(yc), bool); te[idxr[gi == k]] = True; P[te] = fit_apply(ok & ~te, te, sPr, sAr, ext, um, True)[0]
            A[r] = np.abs(P[idxr] - yc[idxr]) / yc[idxr] * 100
        APE[v] = A; print(v, round(float(A.mean()), 4), flush=True)
    u, inv = np.unique(grp[idxr], return_inverse=True); rng = np.random.default_rng(0); B = [rng.integers(0, len(u), len(u)) for _ in range(2000)]
    def cmp(A, R_, m):
        d = A.mean(0) - R_.mean(0); dm = np.where(m, d, 0.0); s = np.bincount(inv, dm, len(u)); n = np.bincount(inv, m.astype(float), len(u))
        bs = np.array([s[k].sum() / max(n[k].sum(), 1) for k in B]); pr = A[:, m].mean(1) - R_[:, m].mean(1)
        return dict(delta=round(float(pr.mean()), 4), sd_repeats=round(float(pr.std()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], share_lower=round(float(np.mean(bs < 0)), 3))
    exm = ~EX[idxr]; SUBi = {'all': np.ones(len(idxr), bool), 'folds1_4': fold[idxr] != 0, 'folds1_3': ~np.isin(fold[idxr], [0, 4]), 'folds0_4': np.isin(fold[idxr], [0, 4])}
    for v, A in APE.items():
        r_ = dict(mean_cd=round(float(A.mean()), 4))
        for refn in [r_n for r_n in ('tierp', 'p2', 'vA46_p2', 'vAboth_p2') if r_n in APE]:
            for sn, m in SUBi.items():
                c_ = cmp(A, APE[refn], m); c_['delta_ex100'] = cmp(A, APE[refn], m & exm)['delta']; r_[f'vs_{refn}_{sn}'] = c_
        res[v] = r_
print(json.dumps(res, indent=1)); json.dump(res, open(LANE / f'reads/heavy_final_{MODE}{TAG}_20261114.json', 'w'), indent=1, default=str)
