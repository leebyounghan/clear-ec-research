"""Interim read of a heavy-member arm (coordinator 01:37 and 03:27 EST 11-14). SLOT=V: ViT-L encoder LR 2e-5 (arms/sweep_vitl_elr2e5_f{0,4}_s46),
plain = the plain ViT-L s46 run of the fold, yardstick = the plain ViT-L s47 run of the fold. SLOT=A: recipe A s46 encoder LR 2e-5
(arms/sweep_ra_elr2e5_f{0,4}_s46), plain = plain recipe A s46 of the fold (C.RECIPE), yardstick = plain s47 and s48. All members as
exp(mean log single view over EPOCHS (env, e.g. 8-10)). Reads, folds 0 and 4: (1) member CD / CV / HEX per fold, arm and yardstick against
plain, donor-clustered paired bootstrap; (2) kept-constants stack per fold: tier-p CD constants fitted on the OTHER fold with the incumbent
members, applied with the slot's s46 half replaced by the arm, by plain, by the yardstick (same single-view form); (3) repeated grouped
10-fold stack read on the rows of folds 0 + 4 (groups slide_id + old-format serial-gap-1 pairs, 20 repeats, coarse grid, member scalars
strict) with the slot's s46 half = arm / plain / yardstick, and the arm as an extra member. Writes reads/interim_<SLOT>_<EPOCHS>_20261114.json."""
import os, re, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
SLOT = os.environ['SLOT']; e0, e1 = [int(v) for v in os.environ['EPOCHS'].split('-')]; ok = yc > 0; y3 = BP.y.astype(float); F04 = np.isin(fold, [0, 4])
def vec3(p):
    z = np.load(p); v = np.full((len(yc), 3), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; return v
def sv(d):
    return np.exp(np.mean([np.log(vec3(d / f'predictions_epoch_{e:02d}.npz')) for e in range(e0, e1 + 1)], 0))
def per_fold(dirs):
    o = np.full((len(yc), 3), np.nan)
    for f, d in dirs.items(): v = sv(d); o[fold == f] = v[fold == f]
    return o
if SLOT == 'V':
    ARM = per_fold({f: LANE / f'arms/sweep_vitl_elr2e5_f{f}_s46/raw/runs/fold{f}/dinov3_l16t' for f in (0, 4)})
    PL = per_fold({f: X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46/raw/runs/fold{f}/dinov3_l16t' for f in (0, 4)})
    YD = {'s47': per_fold({f: LANE / f'arms/vitl16_f{f}_s47/raw/runs/fold{f}/dinov3_l16t' for f in (0, 4)})}
else:
    ARM = per_fold({f: LANE / f'arms/sweep_ra_elr2e5_f{f}_s46/raw/runs/fold{f}/cpdino_b8' for f in (0, 4)})
    PL = per_fold({f: X / C.RECIPE[46][f] / 'runs' / f'fold{f}' / 'cpdino_vitb' for f in (0, 4)})
    YD = {f's{s}': per_fold({f: X / C.RECIPE[s][f] / 'runs' / f'fold{f}' / 'cpdino_vitb' for f in (0, 4)}) for s in (47, 48)}
SCALE_CD = 0.9970446110788349; P1raw = R.P1 / SCALE_CD; A46 = R.seeds[46][:, 0]; A47 = R.seeds[47][:, 0]
D46 = R.member('dinov3_s46_tta'); D47 = R.member('dinov3_s47_tta'); Dm = np.exp(0.5 * np.log(D46) + 0.5 * np.log(D47))
def vit(sd):
    o = np.full(len(yc), np.nan)
    for f in range(5):
        r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; o[fold == f] = p[fold == f]
    return o
V46 = vit('s46'); V47 = vit('s47'); gm = lambda a, b: np.exp(0.5 * np.log(a) + 0.5 * np.log(b))
def members(s46_half):   # returns (Araw, V) with the slot's s46 half replaced (s46_half = CD column array or None for the incumbent)
    if SLOT == 'V': return 0.5 * (A46 + A47), (gm(V46, V47) if s46_half is None else gm(s46_half, V47))
    return (0.5 * (A46 + A47) if s46_half is None else 0.5 * (s46_half + A47)), gm(V46, V47)
def wsc(p, tr): k = tr & ok; return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))
def stretch_fit(lq, y):
    mu = lq.mean(); best = None
    for g in N.GRID:
        q = np.exp(g * (lq - mu) + mu); c = C.wmed(y / q, q / y); m = np.mean(np.abs(c * q - y) / y)
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]
def apply_st(lq, p): g, c, mu = p; return c * np.exp(g * (lq - mu) + mu)
def fit(tr, sPv, sAv, Araw, V, extra=None, coarse=False):
    L = np.stack([np.log(P1raw * sPv), np.log(Araw * sAv), np.log(Dm), np.log(V)] + ([np.log(extra)] if extra is not None else []), 1)
    Aw = (0.3, 0.4, 0.5, 0.6, 0.7) if coarse else np.round(np.arange(0.25, 0.751, 0.05), 3); Ow = (0, 0.1, 0.2, 0.3, 0.4) if coarse else np.round(np.arange(0, 0.401, 0.05), 3)
    grids = []
    for a in Aw:
        for mw in itertools.product(Ow, repeat=L.shape[1] - 2):
            r_ = 1 - a - sum(mw)
            if r_ >= 0.099: grids.append(np.array([r_, a, *mw]))
    trg = tr & good & ok; best = None
    for w in grids:
        st = stretch_fit(L[trg] @ w, yc[trg]); e = np.mean(np.abs(apply_st(L[tr & ok] @ w, st) - yc[tr & ok]) / yc[tr & ok])
        if best is None or e < best[0]: best = (e, w)
    w = best[1]; trm = trg & K.have; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
    for wmk in (0, 0.05, 0.1, 0.15, 0.2):
        qq = np.where(K.have, (1 - wmk) * (L @ w) + wmk * Mc, L @ w); st = stretch_fit(qq[trg], yc[trg]); e = np.mean(np.abs(apply_st(qq[tr & ok], st) - yc[tr & ok]) / yc[tr & ok])
        if b2 is None or e < b2[0]: b2 = (e, wmk, st)
    return dict(w=w, wmk=b2[1], st=b2[2], Mc=Mc)
def apply(P, sPv, sAv, Araw, V, extra=None):
    L = np.stack([np.log(P1raw * sPv), np.log(Araw * sAv), np.log(Dm), np.log(V)] + ([np.log(extra)] if extra is not None else []), 1)
    lq = L @ P['w']; qq = np.where(K.have, (1 - P['wmk']) * lq + P['wmk'] * P['Mc'], lq); return apply_st(qq, P['st'])
def m3(p, rows): return [round(float(np.mean(np.abs(p[rows & (y3[:, j] > 0), j] / y3[rows & (y3[:, j] > 0), j] - 1)) * 100), 4) for j in range(3)]
def paired(a, b, rows, j=0):
    r = rows & (y3[:, j] > 0); d = (np.abs(a[r, j] / y3[r, j] - 1) - np.abs(b[r, j] / y3[r, j] - 1)) * 100; u, inv = np.unique(donors[r], return_inverse=True)
    s = np.bincount(inv, d); n = np.bincount(inv); rng = np.random.default_rng(0); bs = [s[k].sum() / n[k].sum() for k in (rng.integers(0, len(u), len(u)) for _ in range(2000))]
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
out = {'slot': SLOT, 'epochs': f'{e0}-{e1}'}
for f in (0, 4):
    rows = fold == f; o = {'member_arm': m3(ARM, rows), 'member_plain': m3(PL, rows), 'arm_vs_plain_cd': paired(ARM, PL, rows)}
    for k, v in YD.items(): o[f'member_yard_{k}'] = m3(v, rows); o[f'yard_{k}_vs_plain_cd'] = paired(v, PL, rows)
    other = 4 if f == 0 else 0; tr = (fold == other) & ok; sPv = wsc(P1raw, tr); Ai, Vi = members(None); sAv = wsc(Ai, tr); Pk = fit(tr, sPv, sAv, Ai, Vi)
    st = {}
    for name, half in [('incumbent', None), ('arm', ARM[:, 0]), ('plain', PL[:, 0])] + [(f'yard_{k}', v[:, 0]) for k, v in YD.items()]:
        Ar, Vr = members(half); st[name] = round(cd(apply(Pk, sPv, sAv, Ar, Vr), rows), 4)
    st['arm_minus_plain'] = round(st['arm'] - st['plain'], 4)
    for k in YD: st[f'yard_{k}_minus_plain'] = round(st[f'yard_{k}'] - st['plain'], 4)
    o['stack_kept_constants_from_fold'] = other; o['stack'] = st; out[f'fold{f}'] = o
# repeated grouped 10-fold on folds 0 + 4
ids_s = np.array([str(i).strip() for i in ids]); par = {}
def find(a):
    while par.get(a, a) != a: a = par[a]
    return a
dk = np.array([str(d) for d in donors]); old = np.array([not re.match(r'^\d{4}-\d{4}', i) for i in ids_s]); ser = {}
for i in np.where(old)[0]: ser[(ids_s[i].split('-')[1][:2], int(ids_s[i].split('-')[0]))] = i
for (y_, s), i in ser.items():
    j = ser.get((y_, s + 1))
    if j is not None: par[find(dk[i])] = find(dk[j])
grp = np.array([find(d) for d in dk]); idxr = np.where(F04 & ok)[0]; ug = np.unique(grp[idxr])
sPr = np.zeros(len(yc)); sAr = {}
VARS = {'incumbent': (None, None), 'arm_in_half': (ARM[:, 0], None), 'plain_in_half': (PL[:, 0], None), 'arm_extra': (None, ARM[:, 0]), 'plain_extra': (None, PL[:, 0])}
for k, v in YD.items(): VARS[f'yard_{k}_in_half'] = (v[:, 0], None)
for f in range(5): sPr[fold == f] = wsc(P1raw, fold != f)
NR = int(os.environ.get('NR', '20')); APE = {}
for name, (half, extra) in VARS.items():
    Ar, Vr = members(half); sAv = np.zeros(len(yc))
    for f in range(5): sAv[fold == f] = wsc(Ar, fold != f)
    A_ = np.zeros((NR, len(idxr)))
    for r in range(NR):
        rng = np.random.default_rng(3000 + r); lab = dict(zip(ug, rng.permutation(len(ug)) % 10)); gi = np.array([lab[g] for g in grp[idxr]]); P = np.full(len(yc), np.nan)
        for k in range(10):
            te = np.zeros(len(yc), bool); te[idxr[gi == k]] = True; tr = F04 & ok & ~te
            Pk = fit(tr, sPr, sAv, Ar, Vr, extra, coarse=True); P[te] = apply(Pk, sPr, sAv, Ar, Vr, extra)[te]
        A_[r] = np.abs(P[idxr] - yc[idxr]) / yc[idxr] * 100
    APE[name] = A_
u, inv = np.unique(grp[idxr], return_inverse=True); rng = np.random.default_rng(0); B = [rng.integers(0, len(u), len(u)) for _ in range(2000)]
def comp(a, b):
    pr = a.mean(1) - b.mean(1); d = a.mean(0) - b.mean(0); s = np.bincount(inv, d, len(u)); n = np.bincount(inv, None, len(u)); bs = np.array([s[k].sum() / n[k].sum() for k in B])
    return dict(mean_diff=round(float(pr.mean()), 4), sd_repeats=round(float(pr.std()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], share_lower=round(float(np.mean(bs < 0)), 3))
cv = {'mean_cd': {k: round(float(v.mean()), 4) for k, v in APE.items()}, 'arm_in_half_vs_plain_in_half': comp(APE['arm_in_half'], APE['plain_in_half']),
      'arm_in_half_vs_incumbent': comp(APE['arm_in_half'], APE['incumbent']), 'arm_extra_vs_plain_extra': comp(APE['arm_extra'], APE['plain_extra'])}
for k in YD: cv[f'yard_{k}_in_half_vs_plain_in_half'] = comp(APE[f'yard_{k}_in_half'], APE['plain_in_half'])
out['cv10_folds04'] = cv
p = LANE / f'reads/interim_{SLOT}_{e0}-{e1}_20261114.json'; json.dump(out, open(p, 'w'), indent=1, default=str); print(json.dumps(out, indent=1))
