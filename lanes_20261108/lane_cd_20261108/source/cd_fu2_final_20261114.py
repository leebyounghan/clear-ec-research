"""Final read of the DINOv3-B encoder LR 2e-5 follow-up (coordinator 03:58 EST 11-14, item 3), all rows, CPU, true form. Arm = DINOv3-B s46
encoder LR 2e-5, e16-20 four-flip TTA (fold 0: the sweep arm, which was chosen on fold 0; folds 1-4: arms/fu2_elr2e5_f<k>_s46 with their own
fold SSL inits; tta/fu2_elr2e5_f<k>_tta4_e16_20.npz). Tier-p CD (P1, recipe A e20 TTA, D = DINOv3-B s46b/s47 e16-20 TTA seed mean, V = ViT-L
s46/s47 final TTA seed mean, mask slot, stretch) against: a the arm in place of D's s47 half (s46b kept; the swap that is free at inference),
b the arm in place of D's s46b half, c the arm as an extra member with its own weight. MODE strict: strict five folds (scalars, usual weight
grid, mask 0-0.2, stretch on the other four folds). MODE cv10: repeated grouped 10-fold at the stack layer (groups slide_id + old-format
serial-gap-1 pairs; 20 repeats; coarse grid A 0.35 / 0.45 / 0.55, other members 0-0.3 step 0.1; member scalars strict) with donor-clustered
paired bootstrap. Both report folds 1-4 apart from fold 0 and the same without the 11 rows above 100 % APE in the t1 OOF. Plus the member
table (arm vs s46b / s47 per fold). Writes reads/fu2_final_<mode>_20261114.json. Machinery is cd_dropladder_20261114.py's."""
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
ARM = R.member(os.environ.get('ARMKEY', 'fu2_elr2e5_tta')); TAG = os.environ.get('OUTTAG', ''); assert np.isfinite(ARM[ok]).all(), 'arm TTA missing on some rows'
VARS = {'tierp': ([gm(D46, D47), gm(V46, V47)], True), 'a_arm_for_D_s47': ([gm(D46, ARM), gm(V46, V47)], True),
        'b_arm_for_D_s46b': ([gm(ARM, D47), gm(V46, V47)], True), 'c_arm_extra': ([gm(D46, D47), gm(V46, V47), ARM], True)}
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
res = {'mode': MODE, 'ex100_rows': int(EX.sum()), 'member': {}}
F14 = ok & (fold != 0)
for nm_, ref in (('vs_s46b', D46), ('vs_s47', D47)):
    res['member'][nm_] = {f'fold{f}': [round(cd(ARM, fold == f), 4), round(cd(ref, fold == f), 4)] for f in range(5)}
    d_, ci_ = boot(ARM, ref, F14); res['member'][nm_]['folds1-4'] = dict(arm=round(cd(ARM, F14), 4), ref=round(cd(ref, F14), 4), diff=round(d_, 4), ci95=[round(x, 4) for x in ci_])
if MODE == 'strict':
    OOF = {}
    for v, (ext, um) in VARS.items():
        o = np.full(len(yc), np.nan); w = {}
        for f in range(5):
            tr = fold != f; te = fold == f; p, ww = fit_apply(tr, te, wsc(P1raw, tr), wsc(Afin_raw, tr), ext, um, False); o[te] = p; w[f] = ww
        OOF[v] = o; print(v, round(cd(o), 4), flush=True)
    for v, o in OOF.items():
        d, ci = boot(o, OOF['tierp'], ok)
        res[v] = dict(strict=round(cd(o), 4), delta_vs_tierp=round(cd(o) - cd(OOF['tierp']), 4), ci95=[round(x, 4) for x in ci], per_fold_delta={f: round(cd(o, fold == f) - cd(OOF['tierp'], fold == f), 4) for f in range(5)},
                      strict_ex100=round(cd(o, ~EX), 4), delta_ex100=round(cd(o, ~EX) - cd(OOF['tierp'], ~EX), 4),
                      folds1_4=dict(delta=round(cd(o, F14) - cd(OOF['tierp'], F14), 4), ci95=[round(x, 4) for x in boot(o, OOF['tierp'], F14)[1]],
                                    delta_ex100=round(cd(o, F14 & ~EX) - cd(OOF['tierp'], F14 & ~EX), 4)))
    np.savez_compressed(LANE / f'reads/fu2_final_strict_oof{TAG}_20261114.npz', **{k: v for k, v in OOF.items()})
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
    for v, A in APE.items():
        per_rep = A.mean(1) - APE['tierp'].mean(1); d = A.mean(0) - APE['tierp'].mean(0); s = np.bincount(inv, d, len(u)); n = np.bincount(inv, None, len(u))
        bs = np.array([s[k].sum() / n[k].sum() for k in B]); exm = ~EX[idxr]
        res[v] = dict(mean_cd=round(float(A.mean()), 4), delta_vs_tierp=round(float(per_rep.mean()), 4), sd_across_repeats=round(float(per_rep.std()), 4),
                      ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], share_lower=round(float(np.mean(bs < 0)), 3),
                      delta_ex100=round(float(A[:, exm].mean() - APE['tierp'][:, exm].mean()), 4))
        m14 = fold[idxr] != 0; d14 = d.copy(); d14[~m14] = 0; s14 = np.bincount(inv, d14, len(u)); n14 = np.bincount(inv, m14.astype(float), len(u))
        bs14 = np.array([s14[k].sum() / max(n14[k].sum(), 1) for k in B])
        res[v]['folds1_4'] = dict(delta=round(float(A[:, m14].mean() - APE['tierp'][:, m14].mean()), 4), ci95=[round(float(np.percentile(bs14, 2.5)), 4), round(float(np.percentile(bs14, 97.5)), 4)],
                                  share_lower=round(float(np.mean(bs14 < 0)), 3), delta_ex100=round(float(A[:, m14 & exm].mean() - APE['tierp'][:, m14 & exm].mean()), 4))
print(json.dumps(res, indent=1)); json.dump(res, open(LANE / f'reads/fu2_final_{MODE}{TAG}_20261114.json', 'w'), indent=1, default=str)
