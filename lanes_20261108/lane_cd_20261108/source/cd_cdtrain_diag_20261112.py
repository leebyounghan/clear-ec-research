"""D-CDTRAIN-003 diagnostics (coordinator 11-12, information only): why a CD-only recipe A s46 arm that is better alone does not help the CD stack.
Folds 0+4, incumbent recomputed in the same run. ARM=<root prefix under lane_cd_20261108/arms, e.g. cdonly_elr3e5_ra> (fold roots ARM_f{0,4}_s46).
 1. arm in place of plain recipe A s46 with the incumbent's parameters kept (P1/A scalars, weights, mask, stretch fit on the other fold with plain
    members): per fold and both.
 2. arm added next to plain s46 as an extra member (its own log column, weight grid 0-0.4), cross-fit: weight and delta per fold.
 3. log-residual correlations on folds 0+4 (arm, plain s46, P1, ViT-L, the incumbent stack) and member-alone CD by label decile, arm vs plain."""
import os, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
ARM = os.environ['ARM']; rows = np.isin(fold, [0, 4]); ok = yc > 0


def vec(p):
    z = np.load(p); v = np.full(len(yc), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]; return v


arm = np.full(len(yc), np.nan)
for f in (0, 4):
    d = X / f'lane_cd_20261108/arms/{ARM}_f{f}_s46/raw/runs/fold{f}/cpdino_b8'
    s = np.exp(0.5 * np.log(vec(d / 'predictions_final_tta.npz')) + 0.5 * np.mean([np.log(vec(d / f'predictions_epoch_{e:02d}.npz')) for e in range(8, 20)], 0))
    arm[fold == f] = s[fold == f]
P1raw = R.P1 / 0.9970446110788349
A46 = np.exp(0.5 * np.log(R.seeds[46][:, 0]) + 0.5 * np.log(R._rec(46))); A47 = np.exp(0.5 * np.log(R.seeds[47][:, 0]) + 0.5 * np.log(R._rec(47)))
armA = A46.copy(); armA[rows] = arm[rows]
D = np.exp(0.5 * np.log(R.member('dinov3_s46_tta')) + 0.5 * np.log(R.member('dinov3_s47_tta'))); V = R.member('vitl_s4647_tt')


def M(q, m): k = m & ok; return float(np.mean(np.abs(q[k] - yc[k]) / yc[k]) * 100)
def wmed_scalar(p, tr): k = tr & ok; return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))


def fit(tr, Araw, extra=None):
    sP = wmed_scalar(P1raw, tr); sA = wmed_scalar(Araw, tr)
    Lc = [np.log(P1raw * sP), np.log(Araw * sA), np.log(D), np.log(V)] + ([np.log(extra * wmed_scalar(extra, tr))] if extra is not None else [])
    grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Lc) - 2):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    best = None
    for wt in grids:
        q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
        if best is None or e < best[0]: best = (e, wt)
    wt = best[1]; q = np.exp(sum(w * l for w, l in zip(wt, Lc)))
    trm = tr & K.have & ok & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
    for wmk in (0, 0.05, 0.1, 0.15, 0.2):
        qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
        if b2 is None or e < b2[0]: b2 = (e, wmk, p2)
    return dict(sP=sP, sA=sA, wt=wt, wmk=b2[1], st=b2[2], Mc=Mc, extra_scale=(wmed_scalar(extra, tr) if extra is not None else None))


def apply(P, Araw, extra=None):
    Lc = [np.log(P1raw * P['sP']), np.log(Araw * P['sA']), np.log(D), np.log(V)] + ([np.log(extra * P['extra_scale'])] if extra is not None else [])
    q = np.exp(sum(w * l for w, l in zip(P['wt'], Lc))); q = np.where(K.have, np.exp((1 - P['wmk']) * np.log(q) + P['wmk'] * P['Mc']), q)
    return apply1(q, P['st'])


Ainc = 0.5 * (A46 + A47); Aarm = 0.5 * (armA + A47)
inc = np.full(len(yc), np.nan); kept = np.full(len(yc), np.nan); add = np.full(len(yc), np.nan); info = {}
for f, o in ((0, 4), (4, 0)):
    tr = (fold == o) & ok
    Pi = fit(tr, Ainc); p_inc = apply(Pi, Ainc); p_kept = apply(Pi, Aarm)
    Pa = fit(tr, Ainc, extra=armA); p_add = apply(Pa, Ainc, extra=armA)
    sel = fold == f; inc[sel] = p_inc[sel]; kept[sel] = p_kept[sel]; add[sel] = p_add[sel]
    info[f] = dict(incumbent_weights=Pi['wt'], add_weights=Pa['wt'])
res = {'arm': ARM, 'incumbent_folds04': round(M(inc, rows), 4)}
res['1_in_place_incumbent_params_kept'] = {'both': round(M(kept, rows) - M(inc, rows), 4), **{f'fold{f}': round(M(kept, fold == f) - M(inc, fold == f), 4) for f in (0, 4)}}
res['2_added_as_extra_member'] = {'both': round(M(add, rows) - M(inc, rows), 4), **{f'fold{f}': dict(delta=round(M(add, fold == f) - M(inc, fold == f), 4), weights_P1_A_D_V_arm=info[f]['add_weights']) for f in (0, 4)}}
lr = lambda q: np.log(q / yc)
m = rows & ok
cols = {'arm': lr(armA), 'plain_s46': lr(A46), 'P1': lr(P1raw), 'ViT-L': lr(V), 'DINOv3': lr(D), 'stack_incumbent': lr(inc)}
res['3_residual_corr_folds04'] = {f'{a} vs {b}': round(float(np.corrcoef(cols[a][m], cols[b][m])[0, 1]), 3) for a, b in
                                  (('arm', 'plain_s46'), ('arm', 'P1'), ('arm', 'ViT-L'), ('arm', 'DINOv3'), ('plain_s46', 'P1'), ('plain_s46', 'ViT-L'), ('plain_s46', 'DINOv3'), ('arm', 'stack_incumbent'), ('plain_s46', 'stack_incumbent'))}
res['3_mean_log_residual_folds04'] = {k: round(float(np.mean(v[m])), 4) for k, v in cols.items()}
e = np.percentile(yc[m], np.linspace(0, 100, 11)); e[-1] += 1; b = np.digitize(yc, e) - 1
res['3_member_alone_by_label_decile'] = {f'd{k + 1} [{e[k]:.0f},{e[k + 1]:.0f})': dict(plain=round(M(A46, m & (b == k)), 3), arm=round(M(armA, m & (b == k)), 3),
                                          arm_minus_plain=round(M(armA, m & (b == k)) - M(A46, m & (b == k)), 3)) for k in range(10)}
res['member_alone_folds04'] = dict(plain=round(M(A46, rows), 4), arm=round(M(armA, rows), 4))
print(json.dumps(res, indent=1)); json.dump(res, open(X / f'lane_cd_20261108/reads/cdtrain_diag_{ARM}_20261112.json', 'w'), indent=1)
