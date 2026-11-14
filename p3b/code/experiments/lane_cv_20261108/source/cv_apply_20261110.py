"""CV column at test time (CV lane, 2026-11-10). numpy only; reads stored deploy parameters, never labels, fold ids or OOF files.

    import cv_apply_20261110 as CA
    P = CA.load_params()                      # deploy/cv_deploy_params.json + deploy/cv_ridge_params.npz
    cv = CA.apply(P, m)                       # m: dict of length-n arrays, member values from the full-data models (below)
    r = CA.ridge(P, embs, 'CV' or 'HEX')      # simulation-embedding ridge correction alone (the HEX lane uses the HEX column)

Member values in m (one value per image):
  p1_cv        mean CV of the five Phase-1 members, decoded, before the Phase-1 scalar
  a_raw_cv     mean CV of recipe A seeds 46 + 47, 4-flip TTA at epoch 20, before the recipe A scalar
               (or instead: h_cv = hybrid-B CV = 0.5 A_cv + 0.5 P1_cv x phase1 scalar, and a_cv = a_raw_cv x recipeA_scalar_cv)
  graph_cv     GRAPH ensemble: mean over the 3 graph sets (noctx, pre47, pre48) of [mean over seeds and epoch snapshots 21-25 of the
               network's CV output]; the output is a log residual against H, exactly what graph_order_cv_20261110.py predicts
  emb_s0, emb_s1, emb_s2   sim2_l8 embeddings (n x 128) of the image's graphs for network seeds 0-2 (mean over the 3 graph sets, as
               sim_embed_eval_20261105.embed); rows of NaN where the image has no graph.  Or give ridge_cv, or best_cv (= graph_cv + ridge_cv), directly.
  qplain_cv, qorder_cv     plain and order-loss TTA graph quartets: mean over 4 sets x 3 seeds x epochs 21-25 of the CV output
  dino_logcv   mean over seeds 46, 47, 49 of the mean log CV over epochs 16-20 (DINOv3 B/16, single view)
  orderdino_logcv  mean log CV over epochs 16-20 of the order-loss DINOv3 B/16 seed 46 (CD_LOSS=order; single view)
  fz_std, fz_spectrum, fz_cov_sqrt, fz_autocorr   frozen heads: mean over 3 head seeds x epochs 16-20 of the log CV prediction

Formula (one recipe, the five-fold OOF number 10.1101 used it with per-fold parameters):
  A_cv = a_raw_cv * sA;  B_cv = 0.5 A_cv + 0.5 p1_cv * s1;  H_cv = B_cv (hex6 changes only HEX)
  L = log H_cv + graph_cv + ridge_cv + w_g (qplain_cv - graph_cv) + w_o (qorder_cv - graph_cv) + w_d (dino_logcv - log A_cv)
      + w_s (fz_spectrum - fz_std) + w_c (fz_cov_sqrt - fz_std) + w_a (fz_autocorr - fz_std) + w_od (orderdino_logcv - log A_cv)
  CV = c exp(g (L - mu) + mu)                                  (stretch_nested_20261103.apply_stretch on the CV column)

Per-row OOF inputs with per-fold parameters, for parity tests: deploy/cv_inputs_oof.npz (cv_deploy_fit_20261110.py inputs);
apply_fold(Z, f) applies fold f's parameters to the rows of that table.
usage: python cv_apply_20261110.py check    reproduces the five-fold OOF CV column (out/$CV_OOF.npz, default cv_final_20261110b) with per-fold parameters
       python cv_apply_20261110.py insample applies the deploy parameters to the training rows' deploy members (sanity number only)"""
import json, os, sys
from pathlib import Path
import numpy as np

LANE = Path(__file__).resolve().parents[1]
DEPLOY = LANE / 'deploy'
OOF_NAME = os.environ.get('CV_OOF', 'cv_final_20261110')     # the freeze candidate's stack file in out/ (names, w0..w4, w_all, P)
# stack component -> (member key, reference): term = w * (m[key] - reference); reference 'logA' = log A_cv, 'graph' = graph_cv, 'fz_std' = fz_std
TERMS = {'dino': ('dino_logcv', 'logA'), 'orderdino': ('orderdino_logcv', 'logA'), 'graph:plain': ('qplain_cv', 'graph'), 'graph:order': ('qorder_cv', 'graph'),
         'fz:spectrum': ('fz_spectrum', 'fz_std'), 'fz:cov_sqrt': ('fz_cov_sqrt', 'fz_std'), 'fz:autocorr': ('fz_autocorr', 'fz_std'),
         'orderdinov2': ('orderdinov2_logcv', 'logA'), 'ordervitl': ('ordervitl_logcv', 'logA')}


def canon(name):
    """Stack spec name -> component name used in the params ('graph:tta_...' plain quartet, 'graph:gord_...' order quartet)."""
    if name.startswith('graph:tta_'): return 'graph:plain'
    if name.startswith('graph:gord_'): return 'graph:order'
    return name


def load_params(d=DEPLOY):
    P = json.loads((Path(d) / 'cv_deploy_params.json').read_text())
    with np.load(Path(d) / 'cv_ridge_params.npz') as z: P['ridge'] = {k: z[k] for k in z.files}
    return P


def ridge(P, embs, col='CV'):
    """Mean over embedding seeds of the stored ridge on standardised embeddings; 0 where an image has no embedding."""
    R = P['ridge']; out = None; ns = int(R['n_seeds'])
    for s in range(ns):
        Z = np.asarray(embs[s], dtype=np.float64); ok = np.isfinite(Z).all(1); Zf = np.where(ok[:, None], Z, 0.0)
        r = np.where(ok, ((Zf - R[f'mu_{col}_s{s}']) / R[f'sd_{col}_s{s}']) @ R[f'w_{col}_s{s}'] + float(R[f't0_{col}_s{s}']), 0.0)
        out = r if out is None else out + r
    return out / ns


def pre_log(P, m, w=None, sA=None):
    """log of the CV prediction before the stretch."""
    w = P['weights'] if w is None else w; sA = P['recipeA_scalar_cv'] if sA is None else sA
    if 'h_cv' in m:                                   # already-built H_cv and scaled A_cv (A_cv = a_raw_cv x recipeA_scalar_cv)
        a_cv = np.asarray(m['a_cv'], float); b_cv = np.asarray(m['h_cv'], float)
    else:
        a_cv = np.asarray(m['a_raw_cv'], float) * sA; b_cv = 0.5 * a_cv + 0.5 * np.asarray(m['p1_cv'], float) * P['phase1_scalar_cv']
    g = np.asarray(m['graph_cv'], float)
    if 'ridge_cv' in m: r = np.asarray(m['ridge_cv'], float)
    elif 'best_cv' in m: r = np.asarray(m['best_cv'], float) - g        # BEST = GRAPH + ridge (best pipeline's log correction)
    else: r = ridge(P, [m[f'emb_s{s}'] for s in range(int(P['ridge']['n_seeds']))], 'CV')
    L = np.log(b_cv) + g + r; refs = {'logA': np.log(a_cv), 'graph': g}
    for k, wk in w.items():
        if wk == 0: continue
        key, ref = TERMS[k]; L = L + wk * (np.asarray(m[key], float) - (refs[ref] if ref in refs else np.asarray(m[ref], float)))
    return L


def stretch(L, prm):
    g, c, mu = prm; return c * np.exp(g * (L - mu) + mu)


def apply(P, m):
    return stretch(pre_log(P, m), P['stretch_cv'])


def apply_fold(Z, f, rows=None):
    """Fold f's nested parameters (weights, recipe A scalar, stretch) on rows of deploy/cv_inputs_oof.npz (default: fold f's rows)."""
    rows = (Z['fold'] == f) if rows is None else rows; names = [str(n) for n in Z['names']]
    keys = ['p1_cv', 'a_raw_cv', 'graph_cv', 'ridge_cv', 'fz_std'] + [TERMS[n][0] for n in names]
    m = {k: Z[k][rows] for k in keys}; w = dict(zip(names, [float(v) for v in Z[f'w{f}']]))
    L = pre_log(dict(phase1_scalar_cv=float(Z['phase1_scalar_cv'])), m, w=w, sA=float(Z['sA'][f])); return stretch(L, Z[f'stretch{f}'])


def _check():
    """Per-fold parameters on the OOF member values must give the stored five-fold OOF CV column exactly."""
    sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
    import best_pipeline_20261106 as BP
    import stretch_nested_20261103 as N
    out = np.load(LANE / f'out/{OOF_NAME}.npz', allow_pickle=True); cz = np.load(LANE / f'out/{OOF_NAME}_comps.npz', allow_pickle=True)
    names = [canon(str(n)) for n in out['names']]
    hyb = json.loads((Path('/workspace/clear-ec-research/experiments/recipeA_analysis_20261028/hybrid_oof.json')).read_text())
    A = BP._b['A']; P1 = BP._b['P1']; y, fold, good = BP.y, BP.fold, BP.good; scale1 = float(BP.C.SCALE[1])
    pred = np.full(len(y), np.nan)
    for f in range(5):
        sA = hyb['recipeA_lofo_scalars'][str(f)][1]; a_raw = A[:, 1] / sA; logA = np.log(A[:, 1])
        G = BP.GRAPH[f][:, 1]; Rg = BP.BEST[f][:, 1] - G
        m = dict(p1_cv=P1[:, 1] / scale1, a_raw_cv=a_raw, graph_cv=G, ridge_cv=Rg, fz_std=np.zeros(len(y)))
        for k, n in enumerate(names):
            key, ref = TERMS[n]; c = cz[f'c{k}_f{f}']; m[key] = c + (logA if ref == 'logA' else G if ref == 'graph' else 0.0)
        w = dict(zip(names, [float(v) for v in out[f'w{f}']]))
        Pf = dict(phase1_scalar_cv=scale1)
        L = pre_log(Pf, m, w=w, sA=sA)
        C3 = BP.H.copy(); C3[:, 1] = np.exp(L)                      # only the CV column matters here
        prm = N.fit_stretch(C3, y, (fold != f) & good)[1]
        pred[fold == f] = stretch(L, prm)[fold == f]
    ref = out['P'][:, 1]; d = np.nanmax(np.abs(pred - ref)); ok = (y[:, 1] > 0)
    sc = np.mean(np.abs(pred[ok] - y[ok, 1]) / y[ok, 1]) * 100
    print(json.dumps(dict(check='per-fold parameters on OOF members vs stored OOF CV', max_abs_diff=float(d), cv=round(float(sc), 4))), flush=True)
    return d


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'check': _check()
    elif len(sys.argv) > 1 and sys.argv[1] == 'insample':
        import cv_deploy_fit_20261110 as F; F.insample()
    else: print(__doc__)
