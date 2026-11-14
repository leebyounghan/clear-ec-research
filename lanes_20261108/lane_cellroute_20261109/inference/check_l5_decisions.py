"""Deploy lane checks for the ledger (2026-11-11, framework revision):
(1) decision conformance: params/deploy.json + manifests/deploy.json implement D-CD-001 (CD params = freeze_cd_params_20261111 deploy, D = s46b + s47
    e16-20 TTA), D-CV-001 (CV deploy dir = lane_cv deploy, its cv_deploy_params md5, 6 components, no orderdino), D-HEX-004 (embedded HEX params =
    the current lane file deploy_params_joint_attnp.json, 39 columns incl. attnp5, attnp5 in hex_extra, hex46 graph nets, cvw3 two seeds), D-HEX-005
    (cell-route step = kept version P2 win+full, base chain_joint_attnp);
(2) lesson L5 for the cell-route step: the nested code (source/corr_newbase.py's per-fold fit, lines copied verbatim below) run with NO outer fold
    (training rows = all labelled usable rows) must reproduce params/cellroute_deploy.json (mu, sd, w, c, c0)."""
import sys, json, hashlib
import numpy as np
R = '/workspace/clear-ec-research/experiments/'; L = R + 'lane_cellroute_20261109/'; HERE = L + 'inference/'
sys.path[:0] = [R + 'meta_ceiling_20261103/source', L + 'source']
P = json.load(open(HERE + 'params/deploy.json')); MAN = json.load(open(HERE + 'manifests/deploy.json')); CRP = json.load(open(HERE + 'params/cellroute_deploy.json'))
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
out = {}
cd = json.load(open(R + 'lane_cd_20261108/reads/freeze_cd_params_20261111.json'))
out['D-CD-001'] = dict(cd_params_equal_lane=P['cd'] == cd['deploy'], recipeA_cd_scalar_equal=P['recipeA_cd_scalar'] == cd['recipeA_cd_scalar_deploy'],
                       dino_cd_seeds=[s[0].split('/arms/')[1].split('/')[0] for s in MAN['dino_cd']], dino_cd_epochs=[len(s) for s in MAN['dino_cd']],
                       vitl_seeds=len(MAN['vitl_seeds']), recipeA_seeds=len(MAN['recipeA_cd']), p1_members=len(MAN['p1_members']))
cvd = R + 'lane_cv_20261108/deploy/'; cvp = json.load(open(cvd + 'cv_deploy_params.json'))
out['D-CV-001'] = dict(cv_deploy_dir=P['cv'].get('deploy_dir'), cv_deploy_params_md5=md5(cvd + 'cv_deploy_params.json')[:8], components=cvp['components'],
                       orderdino_weight=cvp['weights'].get('orderdino', 0.0), dino_weight=cvp['weights'].get('dino'))
hx = json.load(open(R + 'lane_hex_20261108/preds/deploy_params_joint_attnp.json'))
out['D-HEX-004'] = dict(embedded_equals_lane_file=P['hex'].get('P') == hx, lane_file_md5=md5(R + 'lane_hex_20261108/preds/deploy_params_joint_attnp.json')[:8], fold=P['hex'].get('fold'),
                        n_weights=len(hx['deploy']['w']), attnp5_column='attnp5(1)' in hx['columns'], hex_extra=list(MAN.get('hex_extra', {})),
                        attnp5_ckpts=len(MAN.get('hex_extra', {}).get('attnp5', {}).get('ckpts', [])), hex_graph_nets=list(MAN['hex_graph_nets']), cvw3_ckpts=len(MAN['cvw3']))
out['D-HEX-005'] = dict(cellroute_base=CRP['base'], feature_names=CRP['names'], alpha=CRP['alpha'])
# ---- L5: corr_newbase.py fold fit, outer fold removed ----
import best_pipeline_20261106 as BP
import ceiling_meta_cpu_20261103 as C
import deploy_hexkonan as DK
zb = np.load(R + CRP['base'].split(' ')[0]); base = zb['final_hex']
assert (zb['idx'].astype(int) == BP.idx.astype(int)).all() and (zb['fold'] == BP.fold).all()
zf = np.load(L + 'feats/cellroute_feats.npz'); NM = list(zf['names']); Fr = zf['F'][:, [NM.index(c) for c in DK.NAMES]][BP.idx.astype(int)]
y, fold, good = BP.y, BP.fold, BP.good; X, m = DK.design(Fr, base); ok = y[:, 2] > 0; res = np.log(np.where(ok, y[:, 2], 1.0)) - np.log(base)
tr = good & m & ok      # corr_newbase.py: tr = (fold != f) & good & m & ok, with no outer fold
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; A = (X[tr] - mu) / sd; w = np.linalg.solve(A.T @ A + 100.0 * np.eye(A.shape[1]), A.T @ (res[tr] - res[tr].mean()))
q = base * np.exp(((X - mu) / sd) @ w); c = C.wmed(y[tr, 2] / q[tr], q[tr] / y[tr, 2]); c0 = C.wmed(y[tr, 2] / base[tr], base[tr] / y[tr, 2])
dif = lambda a, b: float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
out['L5_cellroute'] = dict(train_rows=int(tr.sum()), deploy_train_rows=CRP['train_rows'], max_abs_diff=dict(mu=dif(mu, CRP['mu']), sd=dif(sd, CRP['sd']), w=dif(w, CRP['w']), c=abs(c - CRP['c']), c0=abs(c0 - CRP['c0'])))
out['L5_cellroute']['reproduces'] = out['L5_cellroute']['train_rows'] == CRP['train_rows'] and max(out['L5_cellroute']['max_abs_diff'].values()) == 0.0
# applying: DK.correct with these params on the OOF base equals the corr_newbase formula (q * c / c0 on usable rows)
Pd = DK.correct(base, Fr, CRP); Pn = base.copy(); Pn[m] = (q * c / c0)[m]
out['L5_cellroute']['apply_max_rel_diff'] = float(np.max(np.abs(Pd / Pn - 1)))
print('SUMMARY', json.dumps(out), flush=True); json.dump(out, open(L + 'out/check_l5_decisions.json', 'w'), indent=1)
