"""Verifier (D-PKG-002/003): cell-route end-to-end strict files on HEX t2 / t3 (deploy lane, convention (a) chain and (b) OOF): MAPE of
P / P_kept / base recomputed here; base vs the verifier's own assembly of the HEX strict ladder (reads/hex_ladder_verify_20261112, rerun);
tier p strict mean (CD t1 + CV t1 + cell-route on HEX t2) and the full candidate's, from verified values. Writes reads/verify_cellroute_tiers_20261112.json."""
import sys, json, numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as NS
E = '/workspace/clear-ec-research/experiments/'; y = BP.y.astype(float); fold = BP.fold; ok = y[:, 2] > 0
def m(p): return float(np.mean(np.abs(p[ok] - y[ok, 2]) / y[ok, 2]) * 100)
def assemble(key):
    P = np.full(len(fold), np.nan)
    for f in range(5):
        c = np.load(E + f'lane_cd_20261108/reads/hex_ladder_verify_20261112/lad_f{f}.npz')[key]; C_ = BP.H.copy(); C_[:, 2] = c
        prm = NS.fit_stretch(C_, y, (fold != f) & BP.good); P[fold == f] = NS.apply_stretch(C_[fold == f], prm)[:, 2]
    return P
res = {}
for t, key in (('t2', 't2'), ('t3', 't3x')):
    mine = assemble(key); res[t] = {'hex_strict_verifier': round(m(mine), 5)}
    for conv in ('a_chain_convention', 'b_oof_convention'):
        z = np.load(E + f'lane_cellroute_20261109/out/corr_hex_e2e_strict_{t}_{conv}.npz', allow_pickle=True)
        if 'idx' in z.files: assert (z['idx'] == BP.idx).all()
        res[t][conv] = {k: round(m(z[k]), 5) for k in ('P', 'P_kept', 'base') if k in z.files}
        if 'base' in z.files: res[t][conv]['base_vs_verifier_hex_strict_max_abs'] = float(np.nanmax(np.abs(z['base'] - mine)))
cd_t1, cv_t1, cd_inc, cv_inc, hex_full_e2e = 6.17366, 10.1115, 6.17428, 10.11596, 9.71342
res['tier_p_strict_mean'] = round((cd_t1 + cv_t1 + res['t2']['a_chain_convention']['P']) / 3, 5)
res['full_candidate_strict_mean'] = round((cd_inc + cv_inc + hex_full_e2e) / 3, 5)
print(json.dumps(res, indent=1)); json.dump(res, open(E + 'lane_cd_20261108/reads/verify_cellroute_tiers_20261112.json', 'w'), indent=1)
