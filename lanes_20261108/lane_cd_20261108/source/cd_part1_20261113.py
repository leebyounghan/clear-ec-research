"""PI request (coordinator 22:50 EST 11-13) part 1, CPU reads, strict five folds against tier-p CD (t1 form, 6.1737), same procedure as
cd_tiers_20261112.py (P1 / A scalars, weight grid, mask slot, stretch all fit on the other four folds; extra members 0-0.4).
a. ViT-L slot from the swa6 / ema999 tracks (final.pt four-flip TTA, both seeds, geometric seed mean) instead of raw final.pt TTA.
b. recipe A slot = per seed exp(mean log single view epochs 9-12), arithmetic seed mean (as A), in place of e20 TTA, and as an added slot.
VAR env (comma list): t1, vswa6, vema999, aw912, aw912add. Writes reads/cd_part1_<var>_20261113.json."""
import os, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
SCALE_CD = 0.9970446110788349; P1raw = R.P1 / SCALE_CD; ok = yc > 0
Afin_raw = 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0])
Dm = np.exp(0.5 * np.log(R.member('dinov3_s46_tta')) + 0.5 * np.log(R.member('dinov3_s47_tta')))
def vec(p, key='pred'):
    z = np.load(p); v = np.full(len(yc), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z[key][:, 0]; return v
def vit(sd, track):
    out = np.full(len(yc), np.nan)
    for f in range(5):
        r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'{track}/runs/fold{f}/dinov3_l16t'
        v = vec(r / 'predictions_final_tta.npz'); out[fold == f] = v[fold == f]
    assert np.isfinite(out).all(), (sd, track); return out
V = {t: np.exp(0.5 * np.log(vit('s46', t)) + 0.5 * np.log(vit('s47', t))) for t in ('raw', 'swa6', 'ema999')}
def aw(seed, e0=9, e1=12):
    out = np.full(len(yc), np.nan)
    for f, root in C.RECIPE[seed].items():
        d = X / root / 'runs' / f'fold{f}' / 'cpdino_vitb'; m = np.mean([np.log(vec(d / f'predictions_epoch_{e:02d}.npz')) for e in range(e0, e1 + 1)], 0); out[fold == f] = np.exp(m[fold == f])
    assert np.isfinite(out).all(); return out
Aw = 0.5 * (aw(46) + aw(47))
VARS = {'t1': (Afin_raw, [Dm, V['raw']]), 'vswa6': (Afin_raw, [Dm, V['swa6']]), 'vema999': (Afin_raw, [Dm, V['ema999']]),
        'aw912': (Aw, [Dm, V['raw']]), 'aw912add': (Afin_raw, [Dm, V['raw'], 'AW'])}
def wsc(p, tr): k = tr & ok; return float(C.wmed(yc[k] / p[k], p[k] / yc[k]))
def fit(tr, Araw, ext):
    sP = wsc(P1raw, tr); sA = wsc(Araw, tr); cols = []
    for m in ext: cols.append(np.log(Aw * wsc(Aw, tr)) if isinstance(m, str) else np.log(m))
    Lc = [np.log(P1raw * sP), np.log(Araw * sA)] + cols; grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Lc) - 2):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    best = None
    for wt in grids:
        q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
        if best is None or e < best[0]: best = (e, wt, q)
    q = best[2]; trm = tr & K.have & ok & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
    for wmk in (0, 0.05, 0.1, 0.15, 0.2):
        qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
        if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
    return apply1(b2[2], b2[3]), dict(weights=best[1], mask_w=b2[1])
base = np.load(LANE / 'reads/cd_tier_t1_20261112.npz')['oof_cd']
for v in os.environ['VAR'].split(','):
    Araw, ext = VARS[v]; oof = np.full(len(yc), np.nan); info = {}
    for f in range(5):
        p, w = fit(fold != f, Araw, ext); oof[fold == f] = p[fold == f]; info[f] = w
    d, ci = boot(oof, base, ok)
    res = dict(var=v, strict=round(cd(oof), 4), base_t1=round(cd(base), 4), delta=round(cd(oof) - cd(base), 4), ci95=[round(x, 4) for x in ci],
               per_fold_delta={f: round(cd(oof, fold == f) - cd(base, fold == f), 4) for f in range(5)}, weights=info,
               member_alone={'V_raw': round(cd(V['raw']), 4), 'V_swa6': round(cd(V['swa6']), 4), 'V_ema999': round(cd(V['ema999']), 4), 'A_e20tta': round(cd(Afin_raw * R.sc), 4), 'A_w912_x_lofo': round(cd(Aw * R.sc), 4)})
    print(json.dumps(res), flush=True); json.dump(res, open(LANE / f'reads/cd_part1_{v}_20261113.json', 'w'), indent=1, default=str)
