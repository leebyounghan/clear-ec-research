"""Stratified hex6 verdict (parent requirement 2026-10-31): paired delta of the LOFO-blended hex6 correction
vs base B on (a) black-frame images (black_frac > 0.01) and (b) years 2023-2024, with the shared harness
conventions (ape/official/donors/paired_bootstrap/year_of/image_stats).  Configurations: active (nine + pre-only
s46), frozen nine, nine + three pre-only.  The LOFO blend is reconstructed per held fold exactly as in
cd_estimators_20261028.lofo_blend (a, b least squares; w on the MAPE grid; fallback to base without a feature)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/green_track_20261028/source')
import eval_oof_20261031 as H
import cd_estimators_20261028 as P
D = Path('/workspace/clear-ec-research/experiments/green_track_20261028/deploy')
OUT = Path('/workspace/clear-ec-research/experiments/green_track_20261028/p11_hex6_strata')
CONFIGS = {'active_nine_plus_preonly_s46': 'hex6_blend_params_10_oof_feature.npz', 'frozen_nine': 'oof_feature_9families.npz', 'nine_plus_three_preonly': 'hex6_blend_params_12_oof_feature.npz'}


def lofo_blended_hex(feat_img, ok_img, fold_img, y_img, base_img):
    """feat (ALREADY mean log hex6)/ok/fold/y/base indexed by IMAGE index; returns blended HEX per image (base where no feature)."""
    lf = np.where(np.isfinite(feat_img), feat_img, np.nan)
    have = np.isfinite(lf)
    okk = ok_img & (y_img[:, 2] > 0)
    ly, lb = np.log(np.where(y_img[:, 2] > 0, y_img[:, 2], 1)), np.log(base_img[:, 2])
    out = base_img[:, 2].copy(); fits = {}
    for h in range(5):
        fit = okk & (fold_img != h) & have
        coef = np.polyfit(lf[fit], ly[fit], 1)[::-1]
        est_fit = coef[0]+coef[1]*lf[fit]
        w = float(P.W_GRID[int(np.argmin([P.mape(np.exp((1-ww)*lb[fit]+ww*est_fit), y_img[fit, 2]) for ww in P.W_GRID]))])
        hv = okk & (fold_img == h) & have
        out[hv] = np.exp((1-w)*lb[hv]+w*(coef[0]+coef[1]*lf[hv]))
        fits[h] = dict(a=float(coef[0]), b=float(coef[1]), w=w)
    return out, fits


def main():
    base = H.load_base()
    idx = base['idx']; n = len(idx)
    stats = H.image_stats(base)
    years = np.array([H.year_of(i) for i in base['ids']])
    groups = H.donors(base['ids'])
    y_rows, B_rows = base['y'], base['B']
    e_B = H.ape(B_rows, y_rows)
    strata = {'all': np.ones(n, bool), 'black_frac_gt_0.01': stats['black_frac'] > 0.01, 'black_frac_le_0.01': stats['black_frac'] <= 0.01,
              'years_2023_2024': np.isin(years, [2023, 2024]), 'years_le_2022': years <= 2022, 'years_2018_2022': (years >= 2018) & (years <= 2022), 'years_2000_2009': years <= 2009}
    result = dict(created_utc=P.now(), n_rows=int(n), strata_sizes={k: int(v.sum()) for k, v in strata.items()}, configs={})
    for cname, fname in CONFIGS.items():
        z = np.load(D/fname, allow_pickle=True)
        feat_img, ok_img, fold_img = z['feature'], z['ok'], z['fold']
        # image-indexed arrays for y and base
        y_img = np.zeros((n, 3)); b_img = np.zeros((n, 3)); y_img[idx] = y_rows; b_img[idx] = B_rows
        blended_img, fits = lofo_blended_hex(feat_img, ok_img, fold_img, y_img, b_img)
        pred_rows = B_rows.copy(); pred_rows[:, 2] = blended_img[idx]
        e_P = H.ape(pred_rows, y_rows)
        leaked_rows = ~ok_img[idx]
        rep = {}
        for sname, mask in strata.items():
            rows = mask & ~leaked_rows
            if rows.sum() < 30:
                continue
            pb = H.paired_bootstrap(e_P[rows], e_B[rows], groups[rows])
            oP, oB = H.official(e_P[rows]), H.official(e_B[rows])
            rep[sname] = dict(n=int(rows.sum()), base_mean=round(oB['mean'], 4), base_HEX=round(oB['HEX'], 4), blended_mean=round(oP['mean'], 4), blended_HEX=round(oP['HEX'], 4),
                              delta_mean_pp=round(oP['mean']-oB['mean'], 4), delta_HEX_pp=round(oP['HEX']-oB['HEX'], 4), paired_bootstrap=pb)
            print(f"{cname:30s} {sname:20s} n={rows.sum():5d} B mean {oB['mean']:.4f} -> {oP['mean']:.4f} (d {oP['mean']-oB['mean']:+.4f}; HEX d {oP['HEX']-oB['HEX']:+.4f}) boot d {pb['delta']:+.4f} CI [{pb['ci95'][0]:+.4f},{pb['ci95'][1]:+.4f}] p_better {pb['p_better']:.3f}", flush=True)
        result['configs'][cname] = dict(lofo_fits=fits, strata=rep)
    OUT.mkdir(exist_ok=True)
    (OUT/'result.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
