"""Fold-0 reproduction test of recipe_infer.py (test harness: it reads stored OOF values to compare and to inject stubs; the entry does not).
Layer 1, all fold-0 rows: the entry's combine stage on the stored per-row inputs -> stored OOF CD / HEX chain / HEX with cell-route / CV.
Layer 3, a slice of fold-0 validation frames: raw image -> Recipe('fold0').predict with the fold-0 models; inputs without a builder or weights are
injected from the stored values (listed per frame). Compares every member and the final columns with the stored OOF values. usage: test_fold0.py layer1|layer3 [N]"""
import sys, json, time
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference')
import recipe_infer as RI
import members as M
E = 'experiments/'; HERE = E + 'lane_cellroute_20261109/inference/'
P = json.load(open(HERE + 'params/fold0.json')); CRP = json.load(open(HERE + 'params/cellroute_fold0.json'))
zc = np.load(E + 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz'); zh = np.load(E + 'lane_hex_20261108/preds/hex_inputs_oof.npz'); zv = np.load(E + 'lane_cv_20261108/deploy/cv_inputs_oof.npz', allow_pickle=True)
hexref = np.load(E + 'lane_hex_20261108/preds/chain_joint_attnp.npz')['final_hex']; crref = np.load(E + 'lane_cellroute_20261109/out/corr_hex_on_joint_attnp.npz')['P']
zf = np.load(E + 'lane_cellroute_20261109/feats/cellroute_feats.npz'); NM = list(zf['names']); CRF = zf['F'][:, [NM.index(c) for c in RI.DK.NAMES]]
assert (zc['idx'] == zh['idx']).all() and (zc['idx'] == zv['idx']).all(); idx = zc['idx'].astype(int); fold = zc['fold']
rel = lambda a, b: float(np.nanmax(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1)))
r0 = np.where(fold == 0)[0]; CVK = ['graph_cv', 'ridge_cv', 'qplain_cv', 'qorder_cv', 'fz_std', 'fz_spectrum', 'fz_cov_sqrt', 'fz_autocorr']
zget = lambda z, k, rows: z[k][rows] if k in z.files else np.full(len(rows), np.nan)    # inputs a newer stack dropped
def cv_m(rows, sA, s1):
    a = zv['a_raw_cv'][rows] * sA; return dict(h_cv=0.5 * a + 0.5 * zv['p1_cv'][rows] * s1, a_cv=a, dino_logcv=zget(zv, 'dino_logcv', rows), orderdino_logcv=zget(zv, 'orderdino_logcv', rows), **{k: zv[k][rows] for k in CVK})
if sys.argv[1] == 'layer1':
    cd = RI.CDC.combine(zc['P1'][r0], zc['A'][r0], zc['D'][r0], zc['V'][r0], zc['mask_cd'][r0], zc['mask_n_int'][r0], P['cd'])
    I = {k: zget(zh, k, r0) for k in RI.HA.INPUTS}
    if 'attnp5(1)' in P['hex'].get('P', {}).get('columns', []):      # OOF attnp5 as the HEX lane's chain.cnn5_hex (fold-0 run, epochs 8-12, log mean of HEX)
        rd = E + 'lane_backbone_20261108/runs/attnpool_f0_dinov3b16_s46/raw/runs/fold0/dinov3_b16/'; pos = {int(d): k for k, d in enumerate(idx)}; Ls = []
        for e in range(8, 13):
            q = np.load(rd + f'predictions_epoch_{e:02d}.npz'); p = np.full(len(idx), np.nan); p[[pos[int(i)] for i in q['indices']]] = q['pred'][:, 2]; Ls.append(np.log(p))
        I['attnp5'] = np.exp(np.mean(Ls, 0))[r0]
    hx = RI.HA.hex_any(P['hex']['P'], I, P['hex'].get('fold')) if 'P' in P['hex'] else RI.HA.apply(P['hex']['steps'], P['hex']['stretch'], I); cr = RI.DK.correct(hx, CRF[idx[r0]], CRP)
    cv = RI.CA.apply(P['cv'], cv_m(r0, P['cv']['recipeA_scalar_cv'], P['cv']['phase1_scalar_cv']))
    print(json.dumps(dict(rows=len(r0), CD_max_rel_diff_vs_oof=rel(cd, zc['oof_cd'][r0]), HEX_chain_max_rel_diff_vs_oof=rel(hx, hexref[r0]), HEX_cellroute_max_rel_diff_vs_oof=rel(cr, crref[r0]),
                          CV_max_rel_diff_vs_oof=rel(cv, zv['cv_oof'][r0]))), flush=True)
else:
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 3; MM = np.memmap(M.ROOT + 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
    def vsl(run):
        L = np.mean([np.log(np.load(run + f'predictions_epoch_{e:02d}.npz')['pred']) for e in range(8, 12)], 0); return {int(i): v for i, v in zip(np.load(run + 'predictions_epoch_08.npz')['indices'], L)}
    VS = [vsl(E + 'lane_backbone_20261108/runs/vitl16_avg_f0_s46/raw/runs/fold0/dinov3_l16t/'), vsl(E + 'lane_cd_20261108/arms/vitl16_f0_s47/raw/runs/fold0/dinov3_l16t/')]
    def stored_peaks(det):
        z = np.load(E + f'green_track_20261028/{det}/fold0/fold0_sg2_peaks.npz', allow_pickle=True); pk, of = z['peaks'], z['offsets']; return {int(i): pk[of[k]:of[k + 1]].astype(np.float32) for k, i in enumerate(z['indices'])}
    SP = {d: stored_peaks(d) for d in RI.ALL16[3:]}
    R = RI.Recipe('manifests/fold0.json', 'params/fold0.json'); rows = r0[:: len(r0) // N][:N]; res = []
    for k in rows:
        d = int(idx[k]); M.clear_cache()
        prov = {n: (zh[n][k] if n != 'have' else bool(zh[n][k])) for n in ('f6', 'have', 'best', 'graph', 'g_s10x3', 'g_hex46', 'dino_cvw3')}
        prov.update({n: zv[n][k] for n in CVK}); prov.update(V_single_logs=[VS[0][d], VS[1][d]], peaks_other13={det: SP[det][d] for det in RI.ALL16[3:]})
        o = R.predict(np.asarray(MM[d]), provided=prov, skip=('peaks_other13',)); p = o['parts']
        row = dict(frame=d, seconds=o['seconds'], stubs=o['stubs'], A=rel(p['A'][2], zh['A'][k]), A_cd=rel(p['A_cd'], zc['A'][k]), D=rel(p['D'], zc['D'][k]), V=rel(p['V'], zc['V'][k]),
                   P1_cd=rel(p['P1'][0], zc['P1'][k]), DINO_logcv=rel(p['DINO_logcv'], zv['dino_logcv'][k]), orderdino_logcv=rel(p['orderdino_logcv'], zv['orderdino_logcv'][k]),
                   mask_cd=rel(p['seg17'][5], zc['mask_cd'][k]), prior_h6c=rel(p['prior_h6c'], zh['prior_h6c'][k]), ridge_k3=rel(p['ridge_k3'], zh['ridge_k3'][k]),
                   ridge_fall16=rel(p['ridge_fall16_h6_s15'], zh['ridge_fall16_h6_s15'][k]), img=rel(p['img'], zh['img'][k]), cellroute_feats_full=rel(np.asarray(p['cellroute_feats'])[5:], CRF[d][5:]),
                   CD_final=rel(o['CD'], zc['oof_cd'][k]), HEX_chain=rel(p['hex_chain'], hexref[k]), HEX_final=rel(o['HEX'], crref[k]), CV_final=rel(o['CV'], zv['cv_oof'][k]), cps5=rel(p['cps5'], zh['cps5'][k]))
        res.append(row); print(json.dumps(row), flush=True)
    keys = [q for q in res[0] if q not in ('frame', 'seconds', 'stubs')]
    print('MAX over frames:', json.dumps({q: max(r[q] for r in res) for q in keys}), flush=True)
