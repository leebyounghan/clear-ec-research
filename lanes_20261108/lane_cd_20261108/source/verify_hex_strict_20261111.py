"""Verifier: independent assembly of the HEX lane's strict OOF for D-HEX-004 (39-column joint ridge) from runs/strict/strict_f{f}.npz 'joint_attnp'
with chain.final's rule re-implemented here (fold f's candidate in H's HEX column, stretch fit on folds != f & good), and a check of the deploy
lane's end-to-end strict cell-route files (P, P_kept, base); updated integrated strict mean."""
import sys, json, numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as NS
E = '/workspace/clear-ec-research/experiments/'
y = BP.y.astype(float); fold = BP.fold; ok = y[:, 2] > 0


def m(p): return float(np.mean(np.abs(p[ok] - y[ok, 2]) / y[ok, 2]) * 100)


P = np.full(len(fold), np.nan)
for f in range(5):
    cand = np.load(E + f'lane_hex_20261108/runs/strict/strict_f{f}.npz')['joint_attnp']
    C_ = BP.H.copy(); C_[:, 2] = cand; prm = NS.fit_stretch(C_, y, (fold != f) & BP.good); P[fold == f] = NS.apply_stretch(C_[fold == f], prm)[:, 2]
res = {'hex_strict_assembled_by_verifier': round(m(P), 5), 'per_fold': {f: round(float(np.mean(np.abs(P[(fold == f) & ok] - y[(fold == f) & ok, 2]) / y[(fold == f) & ok, 2]) * 100), 4) for f in range(5)}}
for conv in ('a_chain_convention', 'b_oof_convention'):
    z = np.load(E + f'lane_cellroute_20261109/out/corr_hex_e2e_strict_{conv}.npz', allow_pickle=True)
    assert (z['idx'] == BP.idx).all() if 'idx' in z.files else True
    res[conv] = {k: round(m(z[k]), 5) for k in ('P', 'P_kept', 'base') if k in z.files}
    res[conv]['base_vs_verifier_max_abs'] = float(np.max(np.abs(z['base'] - P))) if 'base' in z.files else None
cd = 6.17428; cv = 10.11596
res['integrated_strict_mean_with_hex_e2e_a'] = round((cd + cv + res['a_chain_convention']['P']) / 3, 5)
res['integrated_strict_mean_with_hex_ridge_only'] = round((cd + cv + res['hex_strict_assembled_by_verifier']) / 3, 5)
print(json.dumps(res, indent=1)); json.dump(res, open(E + 'lane_cd_20261108/reads/verify_hex_strict_20261111.json', 'w'), indent=1)
