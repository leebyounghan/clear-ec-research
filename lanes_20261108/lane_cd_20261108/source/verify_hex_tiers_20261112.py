"""Verifier (D-PKG-002): HEX lane tier sets. Inputs: the HEX lane's strict ladder rerun by the verifier into reads/hex_ladder_verify_20261112/
(strict_ladder.py FOLD OUTDIR, the lane's code, fresh run) and the lane's own ladder files. Assembly here with an independent final-stretch
implementation (fold f's candidate in H's HEX column, stretch fit on folds != f & good). Checks: rerun candidates vs the lane's; strict values vs
strict_ladder.json; the t1 rerun vs the D-HEX-004 strict inputs (runs/strict/strict_f*.npz joint_attnp); the nested OOF values of the tier files;
the L5 parity runs (joint_deploy.py parity per tier, logs). Writes reads/verify_hex_tiers_20261112.json."""
import sys, json, re, numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as NS
E = '/workspace/clear-ec-research/experiments/'; HX = E + 'lane_hex_20261108/'; VD = E + 'lane_cd_20261108/reads/hex_ladder_verify_20261112/'
y = BP.y.astype(float); fold = BP.fold; ok = y[:, 2] > 0
def m(p, r=None): r = ok if r is None else (ok & r); return float(np.mean(np.abs(p[r] - y[r, 2]) / y[r, 2]) * 100)
def assemble(cands):
    P = np.full(len(fold), np.nan)
    for f in range(5):
        C_ = BP.H.copy(); C_[:, 2] = cands[f]; prm = NS.fit_stretch(C_, y, (fold != f) & BP.good); P[fold == f] = NS.apply_stretch(C_[fold == f], prm)[:, 2]
    return P
mine = [np.load(VD + f'lad_f{f}.npz') for f in range(5)]; theirs = [np.load(HX + f'runs/strict_ladder/lad_f{f}.npz') for f in range(5)]
lad = json.load(open(HX + 'runs/strict_ladder/strict_ladder.json')); res = {'variants': {}}
names = {'t1': 't1 (D-HEX-004)', 't2': 't2', 't2_no_attnp5': 'step', 't3': 't3core', 't3_no_fall16': 'step', 't3x': 't3'}
for v in mine[0].files:
    Pm = assemble({f: mine[f][v] for f in range(5)}); Pt = assemble({f: theirs[f][v] for f in range(5)})
    res['variants'][v] = dict(label=names.get(v, v), strict_rerun_here=round(m(Pm), 5), strict_lane_files_assembled_here=round(m(Pt), 5), strict_lane_json=lad[v]['all5'],
                              f04_rerun=round(m(Pm, np.isin(fold, [0, 4])), 5), f04_lane_json=lad[v]['f04'],
                              cand_max_abs_rerun_vs_lane=float(max(np.nanmax(np.abs(mine[f][v] - theirs[f][v])) for f in range(5))))
ref = [np.load(HX + f'runs/strict/strict_f{f}.npz')['joint_attnp'] for f in range(5)]
res['t1_rerun_vs_DHEX004_strict_inputs_max_abs'] = float(max(np.nanmax(np.abs(mine[f]['t1'] - ref[f])) for f in range(5)))
res['DHEX004_strict_assembled_here'] = round(m(assemble(ref)), 5)
for t, stated in (('t2', 9.71991), ('t3', 9.73216), ('t3core', 9.73144)):
    z = np.load(HX + f'preds/chain_joint_{t}.npz'); assert (z['idx'] == BP.idx).all()
    res[f'nested_oof_{t}'] = dict(here=round(m(z['final_hex']), 5), stated=stated)
for t in ('t2', 't3', 't3core'):
    txt = open(E + f'lane_cd_20261108/logs/verify_hex_parity_{t}_20261112.log').read(); j = [l for l in txt.splitlines() if l.startswith('{')]
    res[f'L5_parity_{t}'] = json.loads(j[-1]) if j else txt[-400:]
print(json.dumps(res, indent=1)); json.dump(res, open(E + 'lane_cd_20261108/reads/verify_hex_tiers_20261112.json', 'w'), indent=1)
