"""Backbone lane's synthetic-only DINOv3-B zero-shot CD (pretrained on 50k synthetic specular frames, never shown a real label) in the CD stack,
strictly nested (coordinator 11-11): per outer fold, P1 and recipe A scalars, stack weights, slot calibrations and stretch all from the other four
folds (as cd_strict_nesting_20261111, baseline 6.1743). Variants: (1) synthetic CD in the mask slot instead of cd_mm2 (linear calibration in log,
all rows); (2) both slots (cd_mm2 on rows with n_int>=50, synthetic on all rows), two weights; (3) synthetic CD as an extra log-blend member."""
import os, itertools, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
rows = np.isin(fold, range(5))
z = np.load(X / 'lane_backbone_20261108/data/synth_zeroshot_f0pre_20261111.npz', allow_pickle=True)
assert list(z['columns'])[0] == 'CD_synth'
SY = np.full(len(yc), np.nan); SY[[R.pos[int(i)] for i in z['idx']]] = z['pred'][:, 0]
chk = np.array([str(a).strip() for a in ids])[[R.pos[int(i)] for i in z['idx']]]
assert (chk == np.array([str(a).strip() for a in z['ids']])).all() and np.isfinite(SY).all() and (SY > 0).all()
ok = yc > 0
print('log corr with CD label: synthetic', round(float(np.corrcoef(np.log(SY[ok]), np.log(yc[ok]))[0, 1]), 3),
      '| cd_mm2 (n_int>=50 rows)', round(float(np.corrcoef(K.lm[ok & K.have], np.log(yc[ok & K.have]))[0, 1]), 3), flush=True)
SCALE_CD = 0.9970446110788349
P1raw = R.P1 / SCALE_CD; Asnap_raw = R.A2 / R.sc
D = np.exp(0.5 * np.log(R.member('dinov3_s46_tta')) + 0.5 * np.log(R.member('dinov3_s47_tta'))); V = R.member('vitl_s4647_tt')
LS = np.log(SY); W1 = (0, 0.05, 0.1, 0.15, 0.2); W2 = (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)


def wmed_scalar(p, tr):
    k = tr & (yc > 0); return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))


def blend(Lc, tr):
    grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Lc) - 2):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    best = None
    for wt in grids:
        q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
        if best is None or e < best[0]: best = (e, wt, q)
    return best[2], best[1]


def slots(q, tr, use_mm2, use_syn):
    lq = np.log(q); comps = []
    if use_mm2:
        trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); comps.append((K.have, np.polyval(co, K.lm), W1))
    if use_syn:
        trm = tr & (yc > 0) & good; co = np.polyfit(LS[trm], np.log(yc[trm]), 1); comps.append((np.ones(len(yc), bool), np.polyval(co, LS), W2))
    best = None
    for ws in itertools.product(*[c[2] for c in comps]):
        lqq = lq.copy()
        for (have, Mc, _), w in zip(comps, ws):
            lqq = lqq + np.where(have, w * (Mc - lq), 0.)
        qq = np.exp(lqq); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
        if best is None or e < best[0]: best = (e, ws, qq, p2)
    return apply1(best[2], best[3]), best[1]


out = {k: np.full(len(yc), np.nan) for k in ('base', 'syn_slot', 'both_slots', 'syn_member')}; info = {k: {} for k in out}
for f in range(5):
    tr = fold != f
    Lc = [np.log(P1raw * wmed_scalar(P1raw, tr)), np.log(Asnap_raw * wmed_scalar(Asnap_raw, tr)), np.log(D), np.log(V)]
    q, wt = blend(Lc, tr)
    for key, um, us in (('base', True, False), ('syn_slot', False, True), ('both_slots', True, True)):
        p, ws = slots(q, tr, um, us); out[key][fold == f] = p[fold == f]; info[key][f] = (wt, ws)
    q2, wt2 = blend(Lc + [LS], tr); p, ws = slots(q2, tr, True, False); out['syn_member'][fold == f] = p[fold == f]; info['syn_member'][f] = (wt2, ws)
    print('fold', f, json.dumps({k: info[k][f] for k in info}), flush=True)
rep('strict baseline (v3)', out['base'], REF, rows=rows)
for key, name in (('syn_slot', 'synthetic CD in the mask slot instead of cd_mm2'), ('both_slots', 'cd_mm2 slot + synthetic slot'), ('syn_member', 'synthetic CD as an extra member')):
    rep(name, out[key], REF, rows=rows); rep(f'{name} vs strict baseline', out[key], out['base'], rows=rows)
np.savez_compressed(X / 'lane_cd_20261108/reads/synth_zeroshot_cd_20261111.npz', idx=idx, **out)
