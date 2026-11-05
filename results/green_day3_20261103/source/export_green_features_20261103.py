"""Export of the green features for the parent's submission-2 post-processing search, aligned to
experiments/recipeA_analysis_20261028/hybrid_oof.npz `indices` order (8954 rows)."""
import json, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, 'experiments/green_track_20261028/source'); sys.path.insert(0, 'experiments/eval_harness_20261031/source'); sys.path.insert(0, 'experiments/green_day2_20261101/source')
import eval_oof_20261031 as H
import cd_estimators_20261028 as P
import green_t4_numerics_20261101 as M
import green_day2_cd_features_20261101 as D
OUT = Path('experiments/submission2_search_20261103'); OUT.mkdir(parents=True, exist_ok=True)
y, tta4, baseA, fold, seen, keys, F, aa = P.load(); ok = ~seen; n = len(y)
with np.load(P.E/'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True) as h:
    h_idx = h['indices'].astype(int); o = np.argsort(h_idx); B = h['B_hybrid_w0p5'][o]; ids_h = h['ids']
assert np.array_equal(h_idx[o], np.arange(n))
with np.load('experiments/green_day2_20261101/d2b_active_gate/active_family_features.npz') as z:
    hx = {k.replace('hex6_', ''): z[k] for k in z.files if k.startswith('hex6_')}; cnt = {k.replace('count_', ''): z[k] for k in z.files if k.startswith('count_')}
A = np.log(np.stack([hx[k] for k in M.ACTIVE_ORDER]))
with np.errstate(all='ignore'):
    f_active10 = np.nanmean(A, 0)                                   # mean log hex6 (row order)
n_det = np.nansum(np.stack([cnt[k] for k in M.ACTIVE_ORDER]), 0)
base = H.load_base(); st = H.image_stats(base); black = np.zeros(n); black[base['idx']] = st['black_frac']
framed = (black > 0.01).astype(int)
feat = np.exp(f_active10)
D.STRATA_NAMES = {i: f's{i}' for i in range(10)}
s_plain, pred_plain = D.lofo_strat(y, B, fold, ok, feat, np.zeros(n, int), t=2)
q = np.nanquantile(n_det[ok], [1/3, 2/3]); terc = np.digitize(np.where(np.isfinite(n_det), n_det, np.nanmedian(n_det)), q)
s_d2b, pred_d2b = D.lofo_strat(y, B, fold, ok, feat, framed*3+terc, t=2)
have = ok & np.isfinite(feat) & (y[:, 2] > 0)
print('plain LOFO reproduces: %.4f mean pp (expect -0.0311); d2b: %.4f (expect -0.0374)' % (s_plain['overall']['delta_mean_pp'], s_d2b['overall']['delta_mean_pp']), flush=True)
e2 = {}
p = Path('experiments/green_day3_20261103/e2_cellhead/features.npz')
if p.exists():
    with np.load(p) as z:
        for k in ('sg2_s46_hex_mean_p', 'sg2pre_s46_hex_mean_p', 'sg2_s46_hex_mean_p_all', 'sg2pre_s46_hex_mean_p_all'):
            if k in z.files: e2['e2_'+k] = z[k]
out = dict(indices=h_idx, ids=ids_h, fold=fold[h_idx], f_active10=f_active10[h_idx], f_active10_valid=np.isfinite(f_active10)[h_idx], n_det=n_det[h_idx], n_det_valid=np.isfinite(n_det)[h_idx], black_frac=black[h_idx],
           hex_B=B[h_idx, 2], hex_B_hex6_lofo=pred_plain[h_idx], hex_B_hex6_lofo_valid=have[h_idx], hex_B_hex6_d2b=pred_d2b[h_idx], hex_B_hex6_d2b_valid=have[h_idx],
           lofo_params_plain=json.dumps({h_: {k: v for k, v in s_plain[str(h_)].items() if k in ('a', 'b', 'w')} for h_ in range(5)}), lofo_params_d2b=json.dumps({h_: {k: v for k, v in s_d2b[str(h_)].items() if k in ('a', 'b', 'w')} for h_ in range(5)}),
           d2b_count_tercile_edges=q, seen_leaked=seen[h_idx], **{k: v[h_idx] for k, v in e2.items()}, **{k+'_valid': np.isfinite(v)[h_idx] for k, v in e2.items()},
           note='row order = hybrid_oof.npz indices; f_active10 = mean log hex6 over the active ten (deployed definition, bf16 OOF, fold-clean); hex_B_hex6_* = B HEX replaced by the LOFO log-blend (per held fold a, b, w fit on the other folds; d2b = w per framed x n_det-tercile stratum); rows with *_valid False keep B HEX unchanged (the 20 leaked train-green rows and rows without a feature); E2 features are flagged (in-domain sanity failed).')
np.savez_compressed(OUT/'green_features.npz', **out)
chk = np.load(OUT/'green_features.npz', allow_pickle=True)
print({k: (chk[k].shape, str(chk[k].dtype)) for k in chk.files if k not in ('note', 'lofo_params_plain', 'lofo_params_d2b')}, flush=True)
e0 = H.ape(np.c_[B[:, 0], B[:, 1], B[:, 2]][h_idx], y[h_idx]); e1 = e0.copy(); e1[:, 2] = H.ape(np.c_[B[:, 0], B[:, 1], pred_plain][h_idx], y[h_idx])[:, 2]; m = ok[h_idx]
print('check on exported arrays (ok rows): B mean %.4f -> with hex6 lofo %.4f' % (H.official(e0, m)['mean'], H.official(e1, m)['mean']), flush=True)
print('written', OUT/'green_features.npz')
