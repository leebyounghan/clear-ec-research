"""Cell-route lane for the backbone lane (2026-11-09): nested teacher from label-free cell readers, built exactly like
lane_backbone_20261108/source/make_nested_teacher_20261109.py (same rows, folds, labels, HGB setting; for outer fold f a fold-g training row is
predicted by a model fit on the 3 folds outside {f, g}). Feature sets:
  LF  = strictly label-free: image statistics (4) + black_frac + HEX-lane zero-shot Cellpose-SAM mask statistics (seg_loose F, 17) +
        seg-only Konan statistics (feats/seg_konan_feats.npz, 37). No network trained on CLEAR-EC labels touches these.
  BB  = the backbone lane's 46 columns (includes legible-window detector statistics and detector hex6 / n_det: those detectors were initialised
        from each fold's recipe A regression encoder, so for a fold-g row they come from a network that saw fold f's labels).
  BB+LF.
Reports plain 5-fold OOF MAPE for each set and writes teachers/teacher_nested_{LF,BBLF}_f{f}.npz {indices, pred} for outer folds 0 and 4."""
import json, sys, glob
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
X = Path('/workspace/clear-ec-research/experiments'); R = Path('/workspace/clear-ec-research'); OUT = X / 'lane_cellroute_20261109/teachers'
SPL = {0: 'experiments/ssl_screen_20260911/preflight/fold0.npz', 1: 'experiments/ssl_screen_20260911/preflight/fold1.npz',
       2: 'experiments/overnight_target_20260912/preflight/fold2.npz', 3: 'experiments/overnight_target_20260912/preflight/fold3.npz',
       4: 'experiments/target_fold4_20260912/preflight/fold4.npz'}
with np.load(R / 'datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
    Y = z['Y'].astype(np.float64)
N = len(Y); fold_of = np.full(N, -1)
for f, p in SPL.items():
    with np.load(R / p) as z: fold_of[z['validation'].astype(int)] = f
# backbone lane's 46
lw = np.load(X / 'meta_ceiling_20261103/sweep/legible_window.npz', allow_pickle=True)['F'].astype(np.float64)
g = np.load(X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); gi = g['indices'].astype(int); G = np.full((N, 3), np.nan)
G[gi, 0] = np.where(g['f_active10_valid'], g['f_active10'], np.nan); G[gi, 1] = np.where(g['n_det_valid'], g['n_det'], np.nan); G[gi, 2] = g['black_frac']
s = np.load(X / 'eval_harness_20261031/image_stats.npz', allow_pickle=True); si = s['idx'].astype(int); ST = np.full((N, 4), np.nan)
for j, k in enumerate(('mean', 'contrast', 'black_frac', 'focus')): ST[si, j] = s[k]
BB = np.concatenate([lw, G, ST], 1); BB = BB[:, [j for j in range(BB.shape[1]) if len(np.unique(BB[np.isfinite(BB[:, j]), j])) > 1]]
# label-free cell readers
SF = np.full((N, 17), np.nan)
for fn in sorted(glob.glob(str(X / 'lane_hex_20261108/seg_loose/c[0-9][0-9][0-9].npz'))):
    with np.load(fn) as z: fr, Fz = z['frames'].astype(int), z['F']
    SF[fr] = Fz
SK = np.load(X / 'lane_cellroute_20261109/feats/seg_konan_feats.npz')['F']
LF = np.concatenate([ST, G[:, 2:3], SF, SK], 1); LF = LF[:, [j for j in range(LF.shape[1]) if len(np.unique(LF[np.isfinite(LF[:, j]), j])) > 1]]
SETS = {'BB': BB, 'LF': LF, 'BBLF': np.concatenate([BB, LF], 1)}
ok = np.isfinite(Y).all(1) & (Y > 0).all(1) & (fold_of >= 0); LY = np.log(np.where(Y > 0, Y, np.nan))
print(json.dumps({k: v.shape[1] for k, v in SETS.items()}), flush=True)


def fit_predict(F, train_rows, pred_rows):
    out = np.zeros((len(pred_rows), 3))
    for k in range(3):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=1109)
        m.fit(F[train_rows], LY[train_rows, k]); out[:, k] = m.predict(F[pred_rows])
    return np.exp(out)


mape = lambda p, y: (np.abs(p / y - 1) * 100).mean(0)
res = {}
for nm, F in SETS.items():
    oof = np.full((N, 3), np.nan)
    for f in range(5): oof[ok & (fold_of == f)] = fit_predict(F, np.where(ok & (fold_of != f))[0], np.where(ok & (fold_of == f))[0])
    m = mape(oof[ok], Y[ok]); res[nm] = dict(plain_oof_mape=dict(zip(['CD', 'CV', 'HEX'], m.round(3).tolist())), mean=round(float(m.mean()), 3), features=int(F.shape[1]))
    print(nm, json.dumps(res[nm]), flush=True)
    if nm == 'BB': continue
    for f in (0, 4):
        with np.load(R / SPL[f]) as z: tr = z['train'].astype(int)
        assert (fold_of[tr] != f).all() and (fold_of[tr] >= 0).all()
        T = np.full((N, 3), np.nan)
        for gf in sorted(set(range(5)) - {f}):
            rows = tr[fold_of[tr] == gf]; fit_rows = np.where(ok & (fold_of != f) & (fold_of != gf))[0]
            assert not np.isin(fit_rows, np.where(fold_of == f)[0]).any() and not np.isin(fit_rows, rows).any()
            T[rows] = fit_predict(F, fit_rows, rows)
        assert np.isfinite(T[tr]).all() and (T[tr] > 0).all()
        np.savez_compressed(OUT / f'teacher_nested_{nm}_f{f}.npz', indices=tr, pred=T[tr].astype(np.float32))
        mt = mape(T[tr], Y[tr]); print(nm, 'outer fold', f, 'nested teacher MAPE on training rows', dict(zip(['CD', 'CV', 'HEX'], mt.round(3).tolist())), flush=True)
json.dump(res, open(OUT / 'nested_seg_teacher.json', 'w'), indent=1)
