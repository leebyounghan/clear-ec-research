"""D-CDTRAIN-002 read (CD lane): two arms of recipe A s46 on fold 0, CD loss only, encoder LR 1e-5 (arms/cdonly_ra_f0_s46) and 3e-5
(arms/cdonly_elr3e5_ra_f0_s46), against plain recipe A s46 fold 0 (phase-2 runner, all three losses). One-fold screen (L2).
Per epoch (whatever exists): fold-0 validation CD MAPE (single view, raw) and the run's history columns (online train loss = CD-only loss for the arms,
online train CD MAPE, train-probe CD on 256 training rows). When an arm has its e20 TTA: member alone in snapshot form on fold 0, and in place of
plain s46 in the CD stack: P1 / A scalars, weights, mask and stretch fit on fold 4 with plain members, applied to fold 0 (incumbent in the same run).
Writes reads/cdtrain_read_20261112.json."""
import os, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
f0 = fold == 0; ok = yc > 0
ARMS = {'cdonly_lr1e-5': X / 'lane_cd_20261108/arms/cdonly_ra_f0_s46/raw/runs/fold0/cpdino_b8',
        'cdonly_lr3e-5': X / 'lane_cd_20261108/arms/cdonly_elr3e5_ra_f0_s46/raw/runs/fold0/cpdino_b8'}
PLAIN = X / C.RECIPE[46][0] / 'runs/fold0/cpdino_vitb'


def vec(p):
    z = np.load(p); v = np.full(len(yc), np.nan); key = 'pred' if 'pred' in z.files else 'pred_tta4'; v[[R.pos[int(i)] for i in z['indices']]] = z[key][:, 0]; return v


def m0(q): k = f0 & ok & np.isfinite(q); return float(np.mean(np.abs(q[k] - yc[k]) / yc[k]) * 100)


def table(d):
    H = {}
    if (d / 'history.json').exists():
        for h in json.load(open(d / 'history.json')):
            H[h['epoch']] = h
    rows = []
    for e in range(1, 21):
        pth = d / f'predictions_epoch_{e:02d}.npz'
        if not pth.exists(): continue
        h = H.get(e, {})
        rows.append(dict(epoch=e, val_cd_fold0=round(m0(vec(pth)), 4), train_loss=h.get('online_train_loss'),
                         train_cd_online=(h.get('online_train_mape') or {}).get('CD'), train_cd_probe=(h.get('train_probe_score') or {}).get('CD')))
    return rows


out = {'plain_recipeA_s46_fold0': table(PLAIN)}
for k, d in ARMS.items():
    out[k] = table(d) if d.exists() else []


def snap(d, tta_name):
    singles = [np.log(vec(d / f'predictions_epoch_{e:02d}.npz')) for e in range(8, 20)]
    return np.exp(0.5 * np.log(vec(d / tta_name)) + 0.5 * np.mean(singles, 0))


P1raw = R.P1 / 0.9970446110788349
A46 = np.exp(0.5 * np.log(R.seeds[46][:, 0]) + 0.5 * np.log(R._rec(46))); A47 = np.exp(0.5 * np.log(R.seeds[47][:, 0]) + 0.5 * np.log(R._rec(47)))
D = np.exp(0.5 * np.log(R.member('dinov3_s46_tta')) + 0.5 * np.log(R.member('dinov3_s47_tta'))); V = R.member('vitl_s4647_tt')
tr = (fold == 4) & ok


def wmed_scalar(p): k = tr; return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))


def fit_apply(Araw_fit, Araw_apply):
    sP = wmed_scalar(P1raw); sA = wmed_scalar(Araw_fit)
    L_fit = [np.log(P1raw * sP), np.log(Araw_fit * sA), np.log(D), np.log(V)]; grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=2):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    best = None
    for wt in grids:
        q = np.exp(sum(w * l for w, l in zip(wt, L_fit))); prm = fit_stretch1(q, tr & good); e = cd(apply1(q, prm), tr)
        if best is None or e < best[0]: best = (e, wt)
    wt = best[1]; q = np.exp(sum(w * l for w, l in zip(wt, L_fit)))
    trm = tr & K.have & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
    for wmk in (0, 0.05, 0.1, 0.15, 0.2):
        qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
        if b2 is None or e < b2[0]: b2 = (e, wmk, p2)
    L_app = [np.log(P1raw * sP), np.log(Araw_apply * sA), np.log(D), np.log(V)]; qa = np.exp(sum(w * l for w, l in zip(wt, L_app)))
    qa = np.where(K.have, np.exp((1 - b2[1]) * np.log(qa) + b2[1] * Mc), qa)
    return m0(apply1(qa, b2[2])), dict(weights=wt, mask=b2[1])


inc_cd, inc_info = fit_apply(0.5 * (A46 + A47), 0.5 * (A46 + A47))
out['incumbent_fold0_fit_on_fold4'] = dict(cd=round(inc_cd, 4), **inc_info)
out['plain_s46_snapshot_alone_fold0'] = round(m0(A46), 4)
for k, d in ARMS.items():
    if d.exists() and (d / 'predictions_final_tta.npz').exists() and all((d / f'predictions_epoch_{e:02d}.npz').exists() for e in range(8, 20)):
        arm = A46.copy(); s = snap(d, 'predictions_final_tta.npz'); arm[f0] = s[f0]
        cdv, info = fit_apply(0.5 * (A46 + A47), 0.5 * (arm + A47))
        out[f'{k}_final'] = dict(member_alone_snapshot_fold0=round(m0(arm), 4), stack_fold0=round(cdv, 4), delta_vs_incumbent=round(cdv - inc_cd, 4), **info)
print(json.dumps(out, indent=1)); json.dump(out, open(X / 'lane_cd_20261108/reads/cdtrain_read_20261112.json', 'w'), indent=1)
