"""Transfer check (parent 2026-10-31): apply the deployed hex6 parameters UNCHANGED to the shipped hybrid's HEX on
fold-4 rows (fold4_shipped_base_20261031.npz) and compare with the LOFO fold-4 value on B.  Two parameter sets:
(i) deployed full fit (a -0.2084, b 0.5989, w 0.28; fitted on all rows, so in-sample for fold 4), (ii) the fold-4
LOFO fit (a, b, w from folds 0-3 only; clean).  Feature = the deployed 10-family mean log hex6 (OOF, fold-clean)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/green_track_20261028/source')
import eval_oof_20261031 as H
import cd_estimators_20261028 as P
R = Path('/workspace/clear-ec-research/experiments/recipeA_analysis_20261028')
D = Path('/workspace/clear-ec-research/experiments/green_track_20261028/deploy')
OUT = Path('/workspace/clear-ec-research/experiments/green_track_20261028/p12_fold4_transfer')
z = np.load(R/'fold4_shipped_base_20261031.npz', allow_pickle=True)
idx, ids, y = z['indices'].astype(int), np.array([str(i) for i in z['ids']]), z['target'].astype(float)
bases = {k: z[k].astype(float) for k in ('shipped_hybrid', 'B_hybrid_w0p5', 'A3_3seed', 'historical_six_calibrated')}
f = np.load(D/'hex6_blend_params_10_oof_feature.npz', allow_pickle=True)
feat_img, ok_img, fold_img = f['feature'], f['ok'], f['fold']
params = json.loads((D/'hex6_blend_params.json').read_text())
lofo4 = dict(a=params['lofo']['per_fold_ab'][4][0], b=params['lofo']['per_fold_ab'][4][1], w=params['lofo']['per_fold_w'][4])
rows_ok = ok_img[idx] & (y[:, 2] > 0)
feat = feat_img[idx]; have = np.isfinite(feat)
groups = H.donors(ids)
result = dict(created_utc=P.now(), n_rows=int(len(idx)), n_scored=int(rows_ok.sum()), n_with_feature=int((rows_ok & have).sum()),
              params_deployed={k: params[k] for k in ('a', 'b', 'w')}, params_lofo_fold4=lofo4, bases={})
for bname, base in bases.items():
    e0 = H.ape(base, y)
    row = dict(base_official=H.official(e0[rows_ok]))
    for pname, pr in (('deployed_full_fit', params), ('lofo_fold4_fit', lofo4)):
        pred = base.copy()
        m = rows_ok & have
        pred[m, 2] = np.exp((1-pr['w'])*np.log(base[m, 2])+pr['w']*(pr['a']+pr['b']*feat[m]))
        e1 = H.ape(pred, y)
        pb = H.paired_bootstrap(e1[rows_ok], e0[rows_ok], groups[rows_ok])
        o1, o0 = H.official(e1[rows_ok]), H.official(e0[rows_ok])
        row[pname] = dict(blended_official=o1, delta_mean_pp=round(o1['mean']-o0['mean'], 4), delta_HEX_pp=round(o1['HEX']-o0['HEX'], 4), paired_bootstrap=pb)
        print(f"{bname:26s} {pname:17s} base mean {o0['mean']:.4f} HEX {o0['HEX']:.4f} -> {o1['mean']:.4f} / {o1['HEX']:.4f}  d mean {o1['mean']-o0['mean']:+.4f}  d HEX {o1['HEX']-o0['HEX']:+.4f}  boot CI [{pb['ci95'][0]:+.4f},{pb['ci95'][1]:+.4f}] p {pb['p_better']:.3f}", flush=True)
    result['bases'][bname] = row
OUT.mkdir(exist_ok=True)
(OUT/'result.json').write_text(json.dumps(result, indent=1))
print('params deployed', result['params_deployed'], 'lofo fold4', lofo4, 'rows scored', result['n_scored'], 'with feature', result['n_with_feature'])
