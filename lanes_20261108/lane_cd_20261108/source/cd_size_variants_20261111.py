"""Deploy-size price list (coordinator 11-11): five-fold nested OOF CD of smaller CD variants, CPU only, no new training, same search as the
freeze v3 candidate (6.1741: P1 + recipe A snapshot + DINOv3 seed mean + ViT-L s46/s47 tt + mask). Variants:
 (a) without ViT-L; (b) recipe A at its final epoch only (TTA4 e20, seed mean x LOFO scalar) instead of the e8-19 snapshot blend;
 (c) ViT-L final.pt only (TTA4) instead of e8-11 single + final TTA; (d) = a + b; (e) P1 + recipe A final + mask only.
A price list for the size decision, not a change to the candidate."""
import os, itertools, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
rows = np.isin(fold, range(5))
A_snap = R.A2
A_final = 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0]) * R.sc
D = np.exp(0.5 * np.log(R.member('dinov3_s46_tta')) + 0.5 * np.log(R.member('dinov3_s47_tta')))
V = R.member('vitl_s4647_tt')


def vitl_final(sd):
    out = np.full(len(yc), np.nan)
    for f in range(5):
        r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else X / f'lane_cd_20261108/arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
        out[fold == f] = p[fold == f]
    return out


V_final = np.sqrt(vitl_final('s46') * vitl_final('s47'))
for nm, m in [('A snapshot', A_snap), ('A final', A_final), ('ViT-L tt', V), ('ViT-L final only', V_final)]:
    assert np.isfinite(m).all(), nm
    print(f'{nm} alone CD {cd(m, rows):.3f}', flush=True)


def stackN(A, Ms, name):
    Lc = [np.log(R.P1), np.log(A)] + [np.log(m) for m in Ms]; grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Ms)):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    out = np.full(len(yc), np.nan); info = {}
    for f in range(5):
        tr = fold != f; best = None
        for wt in grids:
            q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
            if best is None or e < best[0]: best = (e, wt, q)
        q = best[2]; trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
        for wmk in (0, 0.05, 0.1, 0.15, 0.2):
            qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
            if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
        out[fold == f] = apply1(b2[2][fold == f], b2[3]); info[f] = (best[1], b2[1])
    rep(name, out, REF, rows=rows, note='weights (P1, A, members..., mask) ' + str(info)); return out


base = stackN(A_snap, [D, V], 'v3 candidate: P1 + A snapshot + DINOv3 seed mean + ViT-L tt + mask')
res = {'base': cd(base, rows)}
for key, A, Ms, name in [('a', A_snap, [D], '(a) without ViT-L'),
                         ('b', A_final, [D, V], '(b) recipe A final epoch only'),
                         ('c', A_snap, [D, V_final], '(c) ViT-L final.pt only'),
                         ('d', A_final, [D], '(d) without ViT-L + recipe A final only'),
                         ('e', A_final, [], '(e) P1 + recipe A final + mask only')]:
    o = stackN(A, Ms, name); rep(f'{name} vs v3 candidate', o, base, rows=rows); res[key] = cd(o, rows)
print(json.dumps({k: round(float(v), 4) for k, v in res.items()}), flush=True)
