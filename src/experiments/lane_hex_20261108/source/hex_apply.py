"""Apply the HEX chain's deploy settings to per-image inputs (HEX lane, 2026-11-09). Test-time entry point for the HEX column.
Inputs per image (dict of N-vectors / N x k arrays, see INPUTS); parameters from preds/deploy_params.json (chain_deploy.py).
  python hex_apply.py oof   builds the inputs from the stored OOF members (each row's own-fold value), writes
                            preds/hex_inputs_oof.npz and preds/hex_apply_oof.npz, and prints HEX over the labelled rows.
                            That number is the parity check for an integrated pipeline fed the same inputs (in-sample for the
                            deploy fits, so lower than the nested OOF 9.738)."""
import os, sys, json
import numpy as np

LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
INPUTS = {'A': 'recipe A HEX (N)', 'P1': 'Phase-1 crop package HEX (N)', 'f6': 'green hex6 feature f_active10 (N)', 'have': 'f6 usable (N bool)',
          'best': 'BP simulation-embedding HEX log correction (N)', 'graph': 'BP graph part HEX log correction (N)',
          'g_s10x3': 'mean HEX log correction of graph nets d64_l5_noctx_s10, d64_l5_pre47_s10, d64_l5_pre48_s10 (N)',
          'g_hex46': 'HEX log correction of the HEX-only graph net hex_pre46_s5, 5 seeds (N)',
          'img': 'image statistics mean, contrast, black_frac, focus (N x 4)', 'prior_h6c': 'cm1.0_h6c from feat_prior.py pre3 (N)',
          'ridge_k3': 'feat_ridge.py k3 features h6_s15, h6_s25, h6_s35, h6_dark, darkness (N x 5)', 'ridge_fall16_h6_s15': 'feat_ridge.py fall16 h6_s15 (N)',
          'seg': 'seg_run.py 17 features of the loose Cellpose-SAM segmentation (N x 17, NaN when not segmented)', 'dino_cvw3': 'DINOv3 B/16 cvw3 HEX (N)',
          'cps5': 'CPSAM 512 multi-target no-SSL HEX, exp of the mean log over epochs 8-12 (N)'}


def lg(v): v = np.asarray(v, float); return np.where(v > 0, np.log(np.where(v > 0, v, 1)), np.nan)


def ridge(p, X, lb, m=None):
    X = np.asarray(X, float).reshape(len(lb), -1).copy()
    if p['feature_median'] is not None:
        med = np.asarray(p['feature_median']); miss = ~np.isfinite(X); X[miss] = np.take(med, np.where(miss)[1])
    Xb = np.c_[X, lb]; Z = (Xb - np.asarray(p['mean'])) / np.asarray(p['sd']); d = Z @ np.asarray(p['w']) + p['intercept']
    return d if m is None else np.where(m, np.nan_to_num(d), 0.0)


def apply(steps, stretch, I):
    """steps: [{'step', 'pick', 'params'}] in chain order (deploy: pick = deploy_pick; nested fold f: the fold's pick and fit)"""
    S = {q['step']: q for q in steps}; n = len(I['A'])
    base = np.exp(0.7 * np.log(I['P1']) + 0.3 * np.log(I['A'])); h = np.asarray(I['have'], bool); h6 = -0.2084 + 0.5989 * np.nan_to_num(I['f6'])
    w0 = S['ap1']['params']['value']; bw = np.exp(w0 * np.log(I['P1']) + (1 - w0) * np.log(I['A']))
    lc = np.log(bw); lc[h] = (1 - 0.28) * lc[h] + 0.28 * h6[h]; lc = lc + I['best']                                  # ap1
    lc = lc + np.where(h, (S['h6w']['params']['value'] - 0.28) * (h6 - np.log(base)), 0.0)                        # h6w
    if S['graphens']['pick'] != 'current': lc = lc + I['g_s10x3'] - I['graph']                                    # graphens (s10x3)
    def rid(nm, X, m=None):
        p = S[nm]['params']; return 0.0 if p.get('kind') == 'none' else ridge(p, X, lc, m)
    img = np.asarray(I['img'], float); Ximg = np.c_[np.log(img[:, 0] + 1), np.log(img[:, 1] + 1), img[:, 2], np.log(img[:, 3] + 1)]
    lc = lc + rid('imgstats', Ximg)                                                                               # imgstats
    lc = lc + rid('prior', lg(I['prior_h6c']))                                                                    # prior (cm1.0_h6c)
    rk = np.asarray(I['ridge_k3'], float); pk = S['ridgeev']['pick']
    Xk = np.c_[lg(rk[:, :4]), rk[:, 4]] if pk.startswith('all_a') else (lg(rk[:, ['h6_s15', 'h6_s25', 'h6_s35', 'h6_dark'].index(pk)]) if pk != 'none' else None)
    lc = lc + (rid('ridgeev', Xk) if Xk is not None else 0.0)                                                      # ridgeev (k3)
    gp = S['gnn:hex_pre46_s5']['params'] if 'gnn:hex_pre46_s5' in S else {}                                       # dropped by the 11-11 seed audit
    if gp.get('kind', '').startswith('mix'): lc = lc + gp['share'] * (I['g_hex46'] - I['graph'])                   # graph net mix
    assert S['ridgeev_fall16']['pick'] in ('h6_s15', 'none'); lc = lc + rid('ridgeev_fall16', lg(I['ridge_fall16_h6_s15']))   # ridgeev_fall16
    F = np.asarray(I['seg'], float).copy(); sp = S['segm']['params']
    if sp.get('kind') != 'none':
        assert S['segm']['pick'] == 'bh_a100.0'; m = np.isfinite(F[:, :15]).all(1) & (np.nan_to_num(F[:, 3]) >= sp['nmin']); F[~np.isfinite(F[:, 15]), 15] = sp['h6_strict_fill']
        G = np.c_[lg(F[:, [0, 1, 2]] + 1e-3), np.log(F[:, 3] + 1), lg(F[:, 4]), lg(F[:, 5]), F[:, 6]]; hist = F[:, 8:15] / (F[:, 8:15].sum(1, keepdims=True) + 1e-9)
        lc = lc + ridge(sp, np.c_[G, hist, lg(F[:, 15] + 1e-3)], lc, m)                                           # segm (segmented frames only)
    if 'cvw3_fix' in S or 'cvw3m_fix' in S: lc = lc + 0.2 * (np.log(I['dino_cvw3']) - np.log(base))            # cvw3 member, share 0.2 (cvw3m: dino_cvw3 = exp mean log over seeds 46, 47)
    if 'mtcnn5_fix' in S: lc = lc + 0.2 * (np.log(I['cps5']) - np.log(base))                                    # CPSAM 512 multi-target no-SSL member, share 0.2
    return stretch['c'] * np.exp(stretch['g'] * (lc - stretch['mu']) + stretch['mu'])


def hex_any(P, I, f=None):
    """test-time entry for either freeze form (2026-11-11): the joint nested ridge (P['kind'] starts with 'joint',
    preds/deploy_params_joint.json, source/joint_deploy.py) or the step chain (preds/deploy_params.json). f = None: deploy fit;
    f = outer fold: that fold's nested fit (parity)."""
    if str(P.get('kind', '')).startswith('joint'):
        import joint_deploy as JD
        return JD.apply_joint(P['deploy'], P['stretch'], I) if f is None else JD.apply_joint(P['nested'][f], P['nested_stretch'][f], I)
    return apply(deploy_steps(P), P['stretch'], I) if f is None else apply(nested_steps(P, f), P['nested_stretch'][f], I)


def deploy_steps(P): return [dict(step=q['step'], pick=q['deploy_pick'], params=q['params']) for q in P['steps']]
def nested_steps(P, f): return [dict(step=q['step'], pick=q['nested'][f]['pick'], params=q['nested'][f]['params']) for q in P['steps']]


SEEDMEAN = False


def oof_inputs():
    sys.path.insert(0, f'{LANE}/source'); import chain as C
    N, fold, BP = C.N, C.fold, C.BP; own = lambda d: np.select([fold == f for f in range(5)], [d[f] for f in range(5)])
    g3 = C.gload(['d64_l5_noctx_s10', 'd64_l5_pre47_s10', 'd64_l5_pre48_s10']); z46 = np.load(f'{LANE}/preds/hex_pre46_s5.npz')
    img = np.load('/workspace/clear-ec-research/experiments/eval_harness_20261031/image_stats.npz'); assert (img['idx'] == C.idx).all()
    zp = np.load(f'{LANE}/feats/prior_pre3.npz'); zk = np.load(f'{LANE}/feats/ridge_k3.npz'); zf = np.load(f'{LANE}/feats/ridge_fall16.npz')
    nf = [str(q) for q in zf['names']]
    I = dict(A=BP._b['A'][:, 2], P1=BP._b['P1'][:, 2], f6=np.asarray(BP.f6, float), have=np.asarray(BP._have, bool),
             best=own({f: BP.BEST[f][:, 2] for f in range(5)}), graph=own({f: BP.GRAPH[f][:, 2] for f in range(5)}), g_s10x3=own(g3),
             g_hex46=own({f: z46[f'f{f}'][:, 2] for f in range(5)}), img=np.c_[img['mean'], img['contrast'], img['black_frac'], img['focus']],
             prior_h6c=zp['F'][C.idx].astype(float)[:, [str(q) for q in zp['names']].index('cm1.0_h6c')], ridge_k3=zk['F'][C.idx].astype(float)[:, :5],
             ridge_fall16_h6_s15=zf['F'][C.idx].astype(float)[:, nf.index('h6_s15')], seg=C.segfeat(f'{LANE}/preds/seg_all.npz')[0],
             dino_cvw3=(np.exp(np.mean([np.log(C.cnn5_hex(r, 'dinov3_b16')) for r in ('lane_cv_20261108/arms/dinov3_cvw3_f{f}_s46', 'lane_hex_20261108/runs/cvw3_f{f}_s47')], 0)) if SEEDMEAN
                        else C.cnn5_hex('lane_cv_20261108/arms/dinov3_cvw3_f{f}_s46', 'dinov3_b16')), cps5=C.cnn5_hex('lane_hex_20261108/runs/mtcps_nossl_c512_f'))
    return I, C


if __name__ == '__main__' and sys.argv[1:2] == ['oof']:
    P = json.load(open(f'{LANE}/preds/' + os.environ.get('DEPLOY_OUT', 'deploy_params.json'))); SEEDMEAN = 'cvw3m_fix' in P['chain']; I, C = oof_inputs(); fold, ok, y = C.fold, C.ok, C.y
    np.savez_compressed(f'{LANE}/preds/hex_inputs_oof.npz', idx=C.idx, fold=fold, **I)
    Hn = np.full(len(fold), np.nan)
    for f in range(5):
        r = fold == f; Hn[r] = apply(nested_steps(P, f), P['nested_stretch'][f], {k: np.asarray(v)[r] for k, v in I.items()})
    Hd = apply(deploy_steps(P), P['stretch'], I)
    ref = np.load(f'{LANE}/preds/' + (sys.argv[2] if len(sys.argv) > 2 else 'chain_seedaudit10.npz'))['final_hex']
    np.savez_compressed(f'{LANE}/preds/hex_apply_oof.npz', idx=C.idx, hex_nested=Hn, hex_deploy_in_sample=Hd)
    mape = lambda H, r: round(float((np.abs(H - y[:, 2]) / y[:, 2] * 100)[ok & r].mean()), 4); allr = np.ones(len(fold), bool)
    print(json.dumps(dict(rows=int(ok.sum()), nested_hex_five_folds=mape(Hn, allr), nested_hex_folds04=mape(Hn, (fold == 0) | (fold == 4)),
                          max_abs_diff_vs_chain_final_hex=float(np.nanmax(np.abs(Hn - ref))), chain_oof=P['oof_hex_five_folds'],
                          in_sample_deploy_params_hex=mape(Hd, allr), nested_folds=[mape(Hn, fold == f) for f in range(5)])))
