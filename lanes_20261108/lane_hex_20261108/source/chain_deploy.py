"""Deploy settings for the HEX chain (HEX lane, 2026-11-09).
Replays the given chain steps exactly as chain.py does (nested; reproduces the OOF value) and, for every step, fits the
deploy setting on all five folds:
  - the option with the lowest HEX over all rows (each row scored by its own fold's nested value), and
  - for ridge steps, the ridge refit on all labelled rows, with the previous step's OOF candidate (each row's own-fold value)
    as the base input.
Also fits the final log-affine stretch on all rows and records every file the chain reads.
Writes preds/deploy_params.json. usage: chain_deploy.py STEP ...   (same step names as chain.py; all must be kept steps)"""
import sys, os, json
import numpy as np
LOADED = []
_np_load = np.load


def _load(p, *a, **k):
    LOADED.append(os.path.abspath(str(p))); return _np_load(p, *a, **k)


np.load = _load
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import chain as C

N, y, fold, good, ok, lg, BP, NS = C.N, C.y, C.fold, C.good, C.ok, C.lg, C.BP, C.NS
CAP = {}
_choose = C.choose


def _cap(options, label):
    CAP['opts'] = options; return _choose(options, label)


C.choose = _cap


def own(cand):
    o = np.zeros(N)
    for f in range(5): o[fold == f] = cand[f][fold == f]
    return o


def ridge_all(o, F, alpha, m=None, rows=None):
    """nested_cor / nested_cor_m fit on rows (default: all labelled rows; nested: rows of the other folds, o = that outer fold's
    candidate); returns the parameters (enough to apply to a test row)"""
    F = np.asarray(F, float).reshape(N, -1); lb = lg(o); tr = good & ok & (m if m is not None else np.ones(N, bool)) & (rows if rows is not None else np.ones(N, bool))
    if m is None:
        X = F.copy(); miss = ~np.isfinite(X); med = np.nanmedian(X[tr], 0); X[miss] = np.take(med, np.where(miss)[1]); X = np.c_[X, lb]
    else:
        med = None; X = np.where(m[:, None], np.c_[F, lb], 0.0)
    t = lg(np.where(ok, y[:, 2], 1)) - lb; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; A = ((X - mu) / sd)[tr]
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (t[tr] - t[tr].mean()))
    return dict(kind='ridge on log residual', alpha=alpha, n_fit=int(tr.sum()), rows='segmented rows only (others unchanged)' if m is not None else 'all rows',
                feature_median=None if med is None else med.tolist(), mean=mu.tolist(), sd=sd.tolist(), w=w.tolist(), intercept=float(t[tr].mean()),
                note='input = [features..., log(previous HEX)]; z = (x - mean) / sd; HEX *= exp(z . w + intercept)')


def seg_inputs():
    F, _ = C.segfeat(f'{C.L.LANE}/preds/seg_all.npz'); m = C.seg_usable(F, 50); F = F.copy(); F[~np.isfinite(F[:, 15]), 15] = np.nanmedian(F[m, 15])
    G = np.c_[lg(F[:, [0, 1, 2]] + 1e-3), np.log(F[:, 3] + 1), lg(F[:, 4]), lg(F[:, 5]), F[:, 6]]
    hist = F[:, 8:15] / (F[:, 8:15].sum(1, keepdims=True) + 1e-9); return dict(all=G, hist=hist, bh=np.c_[G, hist, lg(F[:, 15] + 1e-3)], h6=lg(F[:, 0] + 1e-3)), m, float(np.nanmedian(F[m, 15]))


def deploy_params(step, pick, o, rows=None):
    nm = step; R = dict(rows=rows)
    if nm in ('ap1', 'h6w'): return dict(kind='weight', value=float(pick[1:]))
    if nm == 'graphens':
        names = {'current': 'BP graph part unchanged', 's10x3': ['d64_l5_noctx_s10', 'd64_l5_pre47_s10', 'd64_l5_pre48_s10'],
                 'all5': ['d64_l5_noctx_s10', 'd64_l5_pre47_s10', 'd64_l5_pre48_s10', 'd64_l5_pre47_s3', 'd64_l5_pre48_s3']}[pick]
        return dict(kind='graph-network ensemble replacing the BP graph part (mean of HEX log corrections)', graphs=names)
    if pick == 'none': return dict(kind='none')
    if nm == 'imgstats': return dict(features='log(mean+1), log(contrast+1), black_frac, log(focus+1)', **ridge_all(o, C.imgstats(), float(pick[1:]), **R))
    if nm == 'prior':
        z = np.load(f'{C.L.LANE}/feats/prior_pre3.npz'); nmz = [str(s) for s in z['names']]; j = nmz.index(pick)
        return dict(features=f'log {pick} from feats/prior_pre3.npz', **ridge_all(o, lg(z['F'][C.idx].astype(float)[:, j]), 10.0, **R))
    if nm.startswith('ridgeev'):
        tag = 'k3' if nm == 'ridgeev' else nm.split('_', 1)[1]; z = np.load(f'{C.L.LANE}/feats/ridge_{tag}.npz'); F = z['F'][C.idx].astype(float); nmz = [str(q) for q in z['names']]
        if pick.startswith('all_a'): return dict(features=f'log of {nmz[:4]} and {nmz[4]} from feats/ridge_{tag}.npz', **ridge_all(o, np.c_[lg(F[:, :4]), F[:, 4]], float(pick[5:]), **R))
        return dict(features=f'log {pick} from feats/ridge_{tag}.npz', **ridge_all(o, lg(F[:, nmz.index(pick)]), 10.0, **R))
    if nm.startswith('gnn:'):
        if pick.startswith('mix'): return dict(kind='mix of a graph-network HEX log correction into the graph part', member=f'preds/{nm[4:]}.npz', share=float(pick[3:]),
                                               note='HEX *= exp(share * (member HEX log correction - BP graph part HEX log correction))')
        raise ValueError(f'{nm}: deploy pick {pick} (stack) has no deploy refit here')
    if nm == 'segm':
        X, m, med15 = seg_inputs(); grp, a = pick.split('_a'); feats = X[grp]
        return dict(features=f'segmentation group {grp} (preds/seg_all.npz; usable = all of the first 15 features finite and n_int >= 50; h6_strict NaN -> {med15:.4f})', h6_strict_fill=med15, nmin=50, **ridge_all(o, feats, float(a), m, **R))
    raise ValueError(nm)


if __name__ == '__main__':
    steps = sys.argv[1:]; cand = {f: C.L.HB[f].copy() for f in range(5)}; C.read('best pipeline', cand, cand); out = []
    for s in steps:
        CAP.clear(); prev = cand
        new, picks = (C.step_gnn(cand, s[4:]) if s.startswith('gnn:') else C.STEPS[s](cand)); o = C.read(s, new, cand)
        assert o['d_prev_f04'] < 0 or os.environ.get('CHAIN_FORCE') == '1', f'{s} is not a kept step'   # seed audit 11-11: the chain is fixed by the audit rule, not by the per-step folds 0+4 test
        if 'opts' in CAP:
            E = [float(np.mean(np.abs(C.final(c)[:, 2] - y[:, 2])[ok] / y[ok, 2])) for _, c in CAP['opts']]; j = int(np.argmin(E)); dpick = CAP['opts'][j][0]
            prm = deploy_params(s, dpick, own(prev))
        else:
            dpick = picks[0]; prm = dict(kind='fixed share, no choice', pick=dpick)
        nested = [dict(pick=picks[f], params=(deploy_params(s, picks[f], prev[f], rows=(fold != f)) if 'opts' in CAP else prm)) for f in range(5)]
        out.append(dict(step=s, oof_picks=picks, deploy_pick=dpick, f04=o['f04'], all5=o['all5'], params=prm, nested=nested)); cand = new
        print('   deploy pick', dpick, flush=True)
    of = own(cand); C_ = BP.H.copy(); C_[:, 2] = of; prm = NS.fit_stretch(C_, y, good)[2]
    st = dict(kind='log-affine stretch on HEX, fit on all labelled rows', g=float(prm[0]), c=float(prm[1]), mu=float(prm[2]), note='HEX = c * exp(g * (log HEX - mu) + mu)')
    ins = float(np.mean(np.abs(st['c'] * np.exp(st['g'] * (np.log(of) - st['mu']) + st['mu']) - y[:, 2])[ok] / y[ok, 2]) * 100)
    nst = []
    for f in range(5):
        Cf = BP.H.copy(); Cf[:, 2] = cand[f]; q = NS.fit_stretch(Cf, y, (fold != f) & good)[2]; nst.append(dict(g=float(q[0]), c=float(q[1]), mu=float(q[2])))
    res = dict(chain=steps, nested_stretch=nst, oof_hex_five_folds=out[-1]['all5'], oof_hex_folds04=out[-1]['f04'], steps=out, stretch=st,
               in_sample_hex_with_deploy_stretch=round(ins, 4), files_read=sorted(set(LOADED)))
    json.dump(res, open(f'{C.L.LANE}/preds/' + os.environ.get('DEPLOY_OUT', 'deploy_params.json'), 'w'), indent=1)
    print(json.dumps(dict(oof=res['oof_hex_five_folds'], f04=res['oof_hex_folds04'], stretch=st, in_sample=res['in_sample_hex_with_deploy_stretch'], picks=[(q['step'], q['deploy_pick']) for q in out])))
    print('files read:'); [print('  ', p) for p in res['files_read']]
