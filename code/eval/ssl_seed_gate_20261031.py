"""SSL-seed gate for submission 2 (written 2026-10-31 before looking at the result).
Reference R  : hybrid with recipe A = {s46, s47, s48} on SSL encoder 1003 (= submission-1 structure)
Candidate C2 : R + s46 on SSL encoder 1004 (four members; what submission 2 would add)
Candidate C1 : {s46, s47, s46-SSL1004} (swap s48; isolates encoder-seed vs regression-seed diversity, report only)
All: four-flip TTA predictions, the held fold's LOFO scalars, hybrid 0.5 * A + 0.5 * Phase-1 five (calibrated).
Gate (pre-registered in the 10-30 plan): C2 - R <= -0.03 on BOTH folds 0 and 4 -> full-data SSL1004 + s46 regression
for submission 2. Also reported on the black-frame stratum (report only).
"""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

X = '/workspace/clear-ec-research/experiments/'
roots = {0: dict(s46='phase2_pooling_replication_20261026/f0_s46_std', s47='phase2_pooling_replication_20261026/f0_s47_std',
                 s48='oof_recipeA_20261027/fold0/regression_std_s48', ssl1004='ssl_seed_sensitivity_20261028/fold0/regression_std_s46_ssl1004'),
         4: dict(s46='phase2_pooling_20261020/arms/std', s47='phase2_pooling_replication_20261026/f4_s47_std',
                 s48='oof_recipeA_20261027/fold4/regression_std_s48', ssl1004='ssl_seed_sensitivity_20261028/fold4/regression_std_s46_ssl1004')}
b = E.load_base(); y = b['y']
scal = json.load(open(X + 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars']
pos = {int(i): n for n, i in enumerate(b['idx'])}
black = E.image_stats(b)['black_frac'] > 0.01
g = E.donors(b['ids'])
out = {}
for f, mem in roots.items():
    rows = np.where(b['fold'] == f)[0]
    preds = {}
    for k, r in mem.items():
        z = np.load(X + r + f'/runs/fold{f}/cpdino_vitb/predictions_tta4.npz')
        p = np.full((len(b['idx']), 3), np.nan)
        for i, v in zip(z['indices'].astype(int), z['pred_tta4']):
            p[pos[int(i)]] = v
        preds[k] = p[rows]
    s = np.array(scal[str(f)])
    P1 = b['P1'][rows]
    def hyb(keys):
        return 0.5 * np.mean([preds[k] for k in keys], 0) * s + 0.5 * P1
    H = dict(R=hyb(['s46', 's47', 's48']), C2=hyb(['s46', 's47', 's48', 'ssl1004']), C1=hyb(['s46', 's47', 'ssl1004']))
    e = {k: E.ape(v, y[rows]) for k, v in H.items()}
    gg = np.unique(g[rows], return_inverse=True)[1]
    bl = black[rows]
    out[f] = dict(n=int(len(rows)), score={k: round(E.official(v)['mean'], 4) for k, v in e.items()},
                  C2_minus_R=E.paired_bootstrap(e['C2'], e['R'], gg), C1_minus_R=E.paired_bootstrap(e['C1'], e['R'], gg),
                  C2_minus_R_black=E.paired_bootstrap(e['C2'][bl], e['R'][bl], np.unique(gg[bl], return_inverse=True)[1]),
                  member_single={k: round(E.official(E.ape(v * s, y[rows]))['mean'], 4) for k, v in preds.items()})
out['gate_pass'] = bool(all(out[f]['C2_minus_R']['delta'] <= -0.03 for f in (0, 4)))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(out, indent=1))
