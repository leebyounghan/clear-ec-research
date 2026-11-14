# copy of cd_tiers_20261112.py writing *_p3_20261114 files (build_cd_p2_20261114.py)
"""D-PKG-002 (coordinator 11-12 15:00): reduced CD deploy tiers, CPU only. Freeze files untouched.
Strict five-fold value per tier with the incumbent D-CD-001 recomputed by the same code (strict 6.1743 expected), and deploy parameters
from the same procedure with no outer fold (L5). Strict procedure = cd_strict_nesting_20261111.py: per outer fold f, P1 and recipe A are
rescaled by weighted-median scalars fit on folds != f, then the log-blend weight grid (A 0.25-0.75, other members 0-0.4, step 0.05,
P1 = remainder >= 0.1), the cd_mm2 mask slot (n_int >= 50, linear calibration in log, weight 0-0.2) and the log-affine stretch, all fit
on folds != f. Deploy = the same fit with every row as training (no outer fold).
TIER (env, comma list): inc = D-CD-001 (P1, A snapshot, D seed mean, V tt); t1 = P1, A e20 TTA only, D seed mean, V final TTA only;
t2 = P1, A e20 TTA only, D seed mean; t3 = P1, A e20 TTA only, mask; t1d / t2d = t1 / t2 with D = s46 alone (information only).
Writes reads/cd_tier_<tier>_20261112.npz (oof + member inputs) and .json (per-fold and deploy params)."""
import os, json, itertools, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
SCALE_CD = 0.9970446110788349
P1raw = R.P1 / SCALE_CD
Afin_raw = 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0])          # recipe A four-flip TTA at epoch 20, arithmetic seed mean, before scalars
Asnap_raw = R.A2 / R.sc                                            # recipe A snapshot (D-CD-001), before scalars
Dm = np.exp(0.5 * np.log(R.member('dinov3_s46_tta')) + 0.5 * np.log(R.member('dinov3_s47_tta')))
D46 = R.member('dinov3_s46_tta')
Vtt = R.member('vitl_s4647_tt')


def vit_final(sd):
    out = np.full(len(yc), np.nan)
    for f in range(5):
        r = (X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46' if (f in (0, 4) and sd == 's46') else LANE / f'arms/vitl16_f{f}_{sd}') / f'raw/runs/fold{f}/dinov3_l16t'
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
        out[fold == f] = p[fold == f]
    return out


Vfin = np.exp(0.5 * np.log(vit_final('s46')) + 0.5 * np.log(vit_final('s47')))   # ViT-L four-flip TTA at final.pt (epoch 12), geometric seed mean
def vit_arm(kind):   # ViT-L encoder LR 2e-5 runs, final.pt four-flip TTA, raw track (p3 / p3b, coordinator 09:16 EST 11-14)
    out = np.full(len(yc), np.nan)
    for f in range(5):
        if kind == 's46': r = LANE / (f'arms/sweep_vitl_elr2e5_f{f}_s46' if f in (0, 4) else f'arms/vitlfu_elr2e5_f{f}_s46') / f'raw/runs/fold{f}/dinov3_l16t'
        else: r = LANE / f'arms/vitl47fu_elr2e5_f{f}_s47/raw/runs/fold{f}/dinov3_l16t'
        if not (r / 'predictions_final_tta.npz').exists(): continue
        z = np.load(r / 'predictions_final_tta.npz'); p = np.full(len(yc), np.nan); p[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, 0]
        out[fold == f] = p[fold == f]
    return out


VA46 = vit_arm('s46'); VA47 = vit_arm('s47')
TIERS = {'inc': (Asnap_raw, Dm, Vtt), 't1': (Afin_raw, Dm, Vfin), 't2': (Afin_raw, Dm, None), 't3': (Afin_raw, None, None),
         't1d': (Afin_raw, D46, Vfin), 't2d': (Afin_raw, D46, None),
         't1d_v46': (Afin_raw, D46, np.exp(0.5 * np.log(VA46) + 0.5 * np.log(vit_final('s47')))), 't1d_vboth': (Afin_raw, D46, np.exp(0.5 * np.log(VA46) + 0.5 * np.log(VA47)))}
mask_cd = K.F[:, K.nm['cd_mm2']]; mask_n = K.F[:, K.nm['n_int']]


def wmed_scalar(p, tr):
    ok = tr & (yc > 0); return float(C.wmed(yc[ok] / p[ok], p[ok] / yc[ok]))


def fit(tr, Araw, Dk, Vk):
    sP = wmed_scalar(P1raw, tr); sA = wmed_scalar(Araw, tr)
    names = ['P1', 'A'] + (['D'] if Dk is not None else []) + (['V'] if Vk is not None else [])
    Lc = [np.log(P1raw * sP), np.log(Araw * sA)] + ([np.log(Dk)] if Dk is not None else []) + ([np.log(Vk)] if Vk is not None else [])
    grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Lc) - 2):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    best = None
    for wt in grids:
        q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
        if best is None or e < best[0]: best = (e, wt, q)
    q = best[2]; trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
    for wmk in (0, 0.05, 0.1, 0.15, 0.2):
        qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
        if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
    w = {k: 0.0 for k in ('P1', 'A', 'D', 'V')}; w.update({n: float(x) for n, x in zip(names, best[1])})
    g, c, mu = b2[3]
    prm = dict(weights=w, mask_min_cells=50, mask_a=float(co[1]), mask_b=float(co[0]), mask_w=float(b2[1]), stretch_g=float(g), stretch_c=float(c), stretch_mu=float(mu),
               P1_scalar_on_raw=sP, p1_cd_multiplier=sP / SCALE_CD, recipeA_cd_scalar=sA,
               weight_on_grid_edge=[n for n, x in zip(names, best[1]) if (n == 'A' and x in (0.25, 0.75)) or (n in ('D', 'V') and x == 0.4)] + (['mask'] if b2[1] == 0.2 else []))
    return prm, apply1(b2[2], b2[3])


def combine_tier(P1in, Ain, Dk, Vk, prm):    # = cd_combine_20261110.combine with absent members (weight 0) left out
    w = prm['weights']; lq = w['P1'] * np.log(P1in) + w['A'] * np.log(Ain)
    if w['D'] > 0: lq = lq + w['D'] * np.log(Dk)
    if w['V'] > 0: lq = lq + w['V'] * np.log(Vk)
    have = np.isfinite(mask_cd) & (mask_cd > 0) & np.isfinite(mask_n) & (mask_n >= prm['mask_min_cells'])
    mc = prm['mask_a'] + prm['mask_b'] * np.log(np.where(have, mask_cd, 1.0)); lq = np.where(have, lq + prm['mask_w'] * (mc - lq), lq)
    return prm['stretch_c'] * np.exp(prm['stretch_g'] * (lq - prm['stretch_mu']) + prm['stretch_mu'])


for t in os.environ['TIER'].split(','):
    Araw, Dk, Vk = TIERS[t]; assert Vk is None or np.isfinite(Vk[yc > 0]).all(), f'{t}: V missing on some rows'; oof = np.full(len(yc), np.nan); per = {}; P1in = np.full(len(yc), np.nan); Ain = np.full(len(yc), np.nan)
    for f in range(5):
        prm, pred = fit(fold != f, Araw, Dk, Vk); m = fold == f; oof[m] = pred[m]; per[str(f)] = prm
        P1in[m] = P1raw[m] * prm['P1_scalar_on_raw']; Ain[m] = Araw[m] * prm['recipeA_cd_scalar']
        print(t, 'fold', f, json.dumps({k: prm[k] for k in ('weights', 'mask_w', 'stretch_g', 'p1_cd_multiplier', 'recipeA_cd_scalar', 'weight_on_grid_edge')}), flush=True)
    rec = np.full(len(yc), np.nan)
    for f in range(5):
        m = fold == f; rec[m] = combine_tier(P1in, Ain, Dk, Vk, per[str(f)])[m]
    dep, pin = fit(np.ones(len(yc), bool), Araw, Dk, Vk)
    pdep = combine_tier(P1raw * dep['P1_scalar_on_raw'], Araw * dep['recipeA_cd_scalar'], Dk, Vk, dep)
    res = dict(tier=t, strict_five_fold_cd=round(cd(oof), 5), per_fold_cd={f: round(cd(oof, fold == f), 4) for f in range(5)}, per_fold=per, deploy=dep,
               check_perfold_params_reproduce_oof_max_rel=float(np.nanmax(np.abs(rec / oof - 1))),
               L5_deploy_params_via_combine_vs_insample_search_max_rel=float(np.nanmax(np.abs(pdep / pin - 1))), insample_cd_deploy=round(cd(pdep), 5))
    print(t, 'RESULT', json.dumps({k: v for k, v in res.items() if k not in ('per_fold',)}), flush=True)
    json.dump(res, open(LANE / f'reads/cd_tier_{t}_p3_20261114.json', 'w'), indent=1)
    np.savez_compressed(LANE / f'reads/cd_tier_{t}_p3_20261114.npz', idx=idx, fold=fold, y_cd=yc, oof_cd=oof, P1_in=P1in, A_in=Ain,
                        D=(Dk if Dk is not None else np.ones(len(yc))), V=(Vk if Vk is not None else np.ones(len(yc))), mask_cd=mask_cd, mask_n_int=mask_n)
