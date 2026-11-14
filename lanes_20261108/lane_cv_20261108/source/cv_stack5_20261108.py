"""CV lane five-fold stack (2026-11-08). CV-column log corrections added to the best pipeline (BEST), weights chosen nested: for each outer fold,
coordinate descent on a grid minimising CV MAPE on the other folds' rows (after the usual stretch), applied to the outer fold. Never chooses on reported rows.
A component is used only if it covers all five folds (or, with --folds 04, folds 0 and 4: weights then come from the other one of the two).
Components (CV column): graph members (any npz with f0..f4 corrections, minus the shipped GRAPH), DINOv3 B/16 (w (log new - log recipe A TTA), seeds averaged
where present), frozen heads (w (lg arm - lg std), own encoders; folds 0/4 from lane GPU 2, 1-3 from this lane).
usage: cv_stack5_20261108.py [--folds 04] [--save NAME] comp_spec ...   comp_spec: graph:<path-or-lane-run>[+...] | dino | fz:<arm> | g10 | vitl | vith | win2x ..."""
import sys, json, os
import numpy as np
from pathlib import Path
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
X = Path('/workspace/clear-ec-research/experiments'); E = BP.E; y = BP.y; fold = BP.fold; pos = {int(i): n for n, i in enumerate(BP.idx)}; A = BP._b['A']; N_ = len(y)
args = sys.argv[1:]; FOLDS = (0, 1, 2, 3, 4); SAVE = None
if '--folds' in args: i = args.index('--folds'); FOLDS = tuple(int(c) for c in args[i + 1]); del args[i:i + 2]
if '--save' in args: i = args.index('--save'); SAVE = args[i + 1]; del args[i:i + 2]
LO = X / 'lane_cv_20261108/graph_runs'; GS = X / 'meta_ceiling_20261103/sweep/graph_sweep'


def graph_comp(spec):
    Z = []
    for q in spec.split('+'):
        p = Path(q if q.endswith('.npz') else q + '.npz'); p = p if p.is_absolute() else (LO / p if (LO / p).exists() else GS / p); Z.append(np.load(p))
    if not all(f'f{f}' in z for z in Z for f in FOLDS): return None
    v = {f: np.mean([z[f'f{f}'][:, 1] for z in Z], 0) - BP.GRAPH[f][:, 1] for f in FOLDS}; return v  # per fold-model vector (all rows)


def last5(root, f, fam='dinov3_b16'):
    run = X / root / 'runs' / f'fold{f}' / fam
    if not (run / 'predictions_epoch_20.npz').exists(): return None
    out = []
    for e in range(16, 21):
        with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
    return np.array([pos[int(i)] for i in out[0][0]]), np.mean([np.log(p) for _, p in out], 0)[:, 1]


def dino_comp():
    v = np.zeros(N_); roots = {0: ['lane_gpu3_20261106/runs/dinov3_b16_f0_s46', 'lane_gpu2_20261106/runs/dinov3_b16_f0_s47'], 4: ['lane_gpu3_20261106/runs/dinov3_b16_f4_s46', 'lane_gpu2_20261106/runs/dinov3_b16_f4_s47']}
    for f in (1, 2, 3): roots[f] = [f'lane_cd_20261108/arms/dinov3_b16_f{f}_s46', f'lane_cd_20261108/arms/dinov3_b16_f{f}_s47']
    for f in range(5): roots[f] += [f'lane_cv_20261108/arms/dinov3_b16_f{f}_s49', f'lane_cv_20261108/arms/dinov3_b16_f{f}_s50'] if os.environ.get('DINO_MORE') else []
    if os.environ.get('DINO_SEEDS'):  # pinned seed list, e.g. 46,47,49 (the freeze candidate)
        want = [int(v) for v in os.environ['DINO_SEEDS'].split(',')]; where = {46: {0: 'lane_gpu3_20261106/runs', 4: 'lane_gpu3_20261106/runs'}, 47: {0: 'lane_gpu2_20261106/runs', 4: 'lane_gpu2_20261106/runs'}}
        roots = {f: [(where.get(sd, {}).get(f) or ('lane_cd_20261108/arms' if sd in (46, 47) else 'lane_cv_20261108/arms')) + f'/dinov3_b16_f{f}_s{sd}' for sd in want] for f in range(5)}
        if os.environ.get('DINO_PLUS_AUG'): roots = {f: roots[f] + [f'lane_cv_20261108/arms/dinov3_aug_f{f}_s{sd}' for sd in (46, 47)] for f in range(5)}
        if os.environ.get('DINO_PLUS_ORDER'): roots = {f: roots[f] + [f'lane_cv_20261108/arms/dinov3_order_f{f}_s46'] for f in range(5)}
    for f in FOLDS:
        got = [q for q in (last5(r, f) for r in roots[f]) if q is not None]
        if not got: return None
        r = got[0][0]; assert all(np.array_equal(r, g[0]) for g in got); v[r] = np.mean([g[1] for g in got], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def fam_comp(fam):
    v = np.zeros(N_)
    for f in FOLDS:
        roots = [f'lane_gpu3_20261106/runs/{fam}_f{f}_s46', f'lane_cd_20261108/arms/{fam}_f{f}_s46', f'lane_gpu2_20261106/runs/{fam}_f{f}_s47', f'lane_cd_20261108/arms/{fam}_f{f}_s47']
        got = [q for q in (last5(r, f, fam) for r in roots) if q is not None]
        if not got: return None
        r = got[0][0]; assert all(np.array_equal(r, g[0]) for g in got); v[r] = np.mean([g[1] for g in got], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def cvwdino_comp():
    v = np.zeros(N_)
    for f in FOLDS:
        g = last5(f'lane_cv_20261108/arms/dinov3_cvw3_f{f}_s46', f)
        if g is None: return None
        v[g[0]] = g[1] - np.log(A[g[0], 1])
    return {f: v for f in FOLDS}


def dinotta_comp(seeds=(46, 47, 48)):
    v = np.zeros(N_)
    for f in FOLDS:
        got = []
        for sd in seeds:
            p = X / f'lane_cd_20261108/tta/dinov3_s{sd}_f{f}_tta4_e16_20.npz'
            if p.exists():
                with np.load(p, allow_pickle=True) as z: got.append((np.array([pos[int(i)] for i in z['indices'].astype(int)]), np.log(z['pred'][:, 1].astype(np.float64))))
        if not got: return None
        r = got[0][0]; assert all(np.array_equal(r, g[0]) for g in got); v[r] = np.mean([g[1] for g in got], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def vitl_comp():
    """ViT-L/16 DINOv3 multi-target member, raw track, log mean of epochs 8-12 (as the CD lane reads it)."""
    v = np.zeros(N_)
    for f in FOLDS:
        run = X / (f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46/raw/runs/fold{f}/dinov3_l16t' if f in (0, 4) else f'lane_cd_20261108/arms/vitl16_f{f}_s46/raw/runs/fold{f}/dinov3_l16t')
        if not (run / 'predictions_epoch_12.npz').exists(): return None
        out = []
        for e in range(8, 13):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        r = np.array([pos[int(i)] for i in out[0][0]]); v[r] = np.mean([np.log(q[:, 1]) for _, q in out], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def vith_comp():
    """ViT-H+/16 member as the CD lane reads it (lane_cd_20261108/reads/vith16_s46_member_preds.npz: idx, fold, pred), log CV minus log recipe A TTA."""
    with np.load(X / 'lane_cd_20261108/reads/vith16_s46_member_preds.npz', allow_pickle=True) as z: idx, fz, pr, have = z['idx'].astype(int), z['fold'].astype(int), z['pred'].astype(np.float64), set(int(q) for q in z['folds'])
    if not set(FOLDS) <= have: return None
    v = np.zeros(N_); r = np.array([pos[int(i)] for i in idx]); m = np.isin(fz, list(FOLDS)) & (pr[:, 1] > 0)
    v[r[m]] = np.log(pr[m, 1]) - np.log(A[r[m], 1])
    return {f: v for f in FOLDS}


def win2x_comp():
    """Backbone lane recipe A window member (win2x), raw track, log mean of epochs 8-12 (as the backbone lane reads it)."""
    v = np.zeros(N_)
    for f in FOLDS:
        run = X / f'lane_backbone_20261108/runs/recipeA_win2x_f{f}_s46/raw/runs/fold{f}/cpdino_b8'
        if not (run / 'predictions_epoch_12.npz').exists(): return None
        out = []
        for e in range(8, 13):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        r = np.array([pos[int(i)] for i in out[0][0]]); v[r] = np.mean([np.log(q[:, 1]) for _, q in out], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def tta_files_comp(patterns):
    """Mean log CV of four-flip TTA member files (any of the patterns, {f} = fold), minus log recipe A TTA."""
    v = np.zeros(N_)
    for f in FOLDS:
        got = []
        for pat in patterns:
            p = X / pat.format(f=f)
            if p.exists():
                with np.load(p, allow_pickle=True) as z: got.append((np.array([pos[int(i)] for i in z['indices'].astype(int)]), np.log(z['pred'][:, 1].astype(np.float64))))
        if not got: return None
        r = got[0][0]; assert all(np.array_equal(r, g[0]) for g in got); v[r] = np.mean([g[1] for g in got], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def mtcps_comp():
    """HEX lane multi-target CPSAM 512 (no SSL), log mean of epochs 8-12."""
    v = np.zeros(N_)
    for f in FOLDS:
        run = X / f'lane_hex_20261108/runs/mtcps_nossl_c512_f{f}/runs/fold{f}/cpsam_c512'
        if not (run / 'predictions_epoch_12.npz').exists(): return None
        out = []
        for e in range(8, 13):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        r = np.array([pos[int(i)] for i in out[0][0]]); v[r] = np.mean([np.log(q[:, 1]) for _, q in out], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def augdino_comp():
    v = np.zeros(N_)
    for f in FOLDS:
        g = last5(f'lane_cv_20261108/arms/dinov3_aug_f{f}_s46', f)
        if g is None: return None
        v[g[0]] = g[1] - np.log(A[g[0], 1])
    return {f: v for f in FOLDS}


def orderdino_comp():
    """Order-loss DINOv3 B/16 (CD_LOSS=order, seed 46), lane_cv arms dinov3_order_f{f}_s46, epochs 16-20 mean log CV minus log recipe A TTA."""
    v = np.zeros(N_); seeds = [int(q) for q in os.environ.get('ORDER_SEEDS', '46').split(',')]   # ORDER_SEEDS=46,47: mean of log CV over seeds
    for f in FOLDS:
        got = [q for q in (last5(f'lane_cv_20261108/arms/dinov3_order_f{f}_s{sd}', f) for sd in seeds) if q is not None]
        if len(got) < len(seeds): return None
        r = got[0][0]; assert all(np.array_equal(r, g[0]) for g in got); v[r] = np.mean([g[1] for g in got], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def orderdinov2_comp():
    """Order-loss DINOv2 B/14 seed 46 (CD_LOSS=order), lane_cv arms dinov2_order_f{f}_s46, epochs 16-20 mean log CV minus log recipe A TTA."""
    v = np.zeros(N_)
    for f in FOLDS:
        g = last5(f'lane_cv_20261108/arms/dinov2_order_f{f}_s46', f, 'dinov2_b14')
        if g is None: return None
        v[g[0]] = g[1] - np.log(A[g[0], 1])
    return {f: v for f in FOLDS}


def ordervitl_comp():
    """Order-loss ViT-L/16 seed 46 (bb_runner_order_cv_20261110), raw track, log mean of epochs 8-12 (as the ViT-L member is read)."""
    v = np.zeros(N_)
    for f in FOLDS:
        run = X / f'lane_cv_20261108/arms/vitl16_order_f{f}_s46/raw/runs/fold{f}/dinov3_l16t'
        if not (run / 'predictions_epoch_12.npz').exists(): return None
        out = []
        for e in range(8, 13):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        r = np.array([pos[int(i)] for i in out[0][0]]); v[r] = np.mean([np.log(q[:, 1]) for _, q in out], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def sd_comp(cd_source):
    """SD member (TARGET_SD=1, lane_cv arms dinov3_sd_f{f}_s46): CV = SD_pred x CD_pred / 1e6, SD = epochs 16-20 mean log of column 1.
    cd_source 'cdlane': CD lane final OOF CD (lane_cd_20261108/reads/freeze_cd_oof_20261109.npz oof_cd); 'own': the member's own CD head (column 0)."""
    v = np.zeros(N_)
    if cd_source == 'cdlane':
        with np.load(X / 'lane_cd_20261108/reads/freeze_cd_oof_20261109.npz', allow_pickle=True) as z:
            assert np.array_equal(z['idx'].astype(int), BP.idx.astype(int)); logcd = np.log(z['oof_cd'].astype(np.float64))
    for f in FOLDS:
        run = X / f'lane_cv_20261108/arms/dinov3_sd_f{f}_s46' / 'runs' / f'fold{f}' / 'dinov3_b16'
        if not (run / 'predictions_epoch_20.npz').exists(): return None
        out = []
        for e in range(16, 21):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        r = np.array([pos[int(i)] for i in out[0][0]]); L = np.mean([np.log(q) for _, q in out], 0)
        lcd = L[:, 0] if cd_source == 'own' else logcd[r]
        v[r] = L[:, 1] + lcd - np.log(1e6) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def ldsdino_comp():
    """LDS DINOv3 B/16 seed 46 (CD_LOSS=lds), lane_cv arms dinov3_lds_f{f}_s46, epochs 16-20 mean log CV minus log recipe A TTA."""
    v = np.zeros(N_)
    for f in FOLDS:
        g = last5(f'lane_cv_20261108/arms/dinov3_lds_f{f}_s46', f)
        if g is None: return None
        v[g[0]] = g[1] - np.log(A[g[0], 1])
    return {f: v for f in FOLDS}


def cmix_comp(track='raw', prefix='cmix'):
    """Backbone lane C-Mixup DINOv3-B seed 46 (runs/cmix_f{f}_dinov3b16_s46), one track, log mean of epochs 8-12 (as the backbone lane reads it)."""
    v = np.zeros(N_)
    for f in FOLDS:
        run = X / f'lane_backbone_20261108/runs/{prefix}_f{f}_dinov3b16_s46/{track}/runs/fold{f}/dinov3_b16'
        if not (run / 'predictions_epoch_12.npz').exists(): return None
        out = []
        for e in range(8, 13):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        r = np.array([pos[int(i)] for i in out[0][0]]); v[r] = np.mean([np.log(q[:, 1]) for _, q in out], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def distdino_comp():
    v = np.zeros(N_)
    for f in FOLDS:
        g = last5(f'lane_cd_20261108/arms/dinov3_dist_f{f}_s46', f)
        if g is None: return None
        v[g[0]] = g[1] - np.log(A[g[0], 1])
    return {f: v for f in FOLDS}


def fz_comp(arm):
    v = np.zeros(N_)
    for f in FOLDS:
        me, l2 = X / 'lane_cv_20261108/reads/own', X / 'lane_gpu2_20261106/reads/own'
        pa = me / f'{arm}_f{f}.npz' if (me / f'{arm}_f{f}.npz').exists() else l2 / f'{arm}_f{f}.npz'; ps = me / f'std_f{f}.npz' if (me / f'std_f{f}.npz').exists() else l2 / f'std_f{f}.npz'
        if not pa.exists() or not ps.exists(): return None
        za, zs = np.load(pa), np.load(ps); assert np.array_equal(za['rows'], zs['rows']) and (fold[za['rows']] == f).all(); v[za['rows']] = za['lg'][:, 1] - zs['lg'][:, 1]
    return {f: v for f in FOLDS}


comps = {}
for s in args:
    if s == 'dino': c = dino_comp()
    elif s == 'cvwdino': c = cvwdino_comp()
    elif s == 'augdino': c = augdino_comp()
    elif s == 'distdino': c = distdino_comp()
    elif s == 'orderdino': c = orderdino_comp()
    elif s.startswith('cmix'): c = cmix_comp(s.split(':')[1] if ':' in s else 'raw')
    elif s.startswith('bbrun:'): c = cmix_comp(s.split(':')[2] if s.count(':') > 1 else 'raw', s.split(':')[1])   # bbrun:<prefix>[:<track>], backbone lane runs/<prefix>_f{f}_dinov3b16_s46
    elif s == 'ldsdino': c = ldsdino_comp()
    elif s in ('sdcd', 'sdown'): c = sd_comp('cdlane' if s == 'sdcd' else 'own')
    elif s == 'orderdinov2': c = orderdinov2_comp()
    elif s == 'ordervitl': c = ordervitl_comp()
    elif s == 'dinotta': c = dinotta_comp()
    elif s == 'vitl': c = vitl_comp()
    elif s == 'mtcps': c = mtcps_comp()
    elif s == 'vith': c = vith_comp()
    elif s == 'win2x': c = win2x_comp()
    elif s == 'cvwtta': c = tta_files_comp(['lane_cv_20261108/out/member_tta/cvw3_s46_f{f}_tta4_e16_20.npz'])
    elif s == 'dinomix': c = tta_files_comp(['lane_cd_20261108/tta/dinov3_s46_f{f}_tta4_e16_20.npz', 'lane_cd_20261108/tta/dinov3_s47_f{f}_tta4_e16_20.npz', 'lane_cd_20261108/tta/dinov3_s48_f{f}_tta4_e16_20.npz', 'lane_cv_20261108/out/member_tta/cvw3_s46_f{f}_tta4_e16_20.npz', 'lane_cv_20261108/out/member_tta/dinov3_s49_f{f}_tta4_e16_20.npz'])
    elif s == 'g10': c = graph_comp('d64_l5_noctx_s10+d64_l5_pre47_s10+d64_l5_pre48_s10')
    elif s.startswith('fz:'): c = fz_comp(s[3:])
    elif s.startswith('fam:'): c = fam_comp(s[4:])
    elif s.startswith('graph:'): c = graph_comp(s[6:])
    else: raise SystemExit('bad spec ' + s)
    print(('ok   ' if c is not None else 'SKIP ') + s, flush=True)
    if c is not None: comps[s] = c
if os.environ.get('STACK_DUMP'):   # 11-11: save the component vectors (CV column, per fold model) for the from-scratch search and stop
    np.savez_compressed(X / f"lane_cv_20261108/out/{os.environ['STACK_DUMP']}.npz", names=np.array(list(comps)), **{f'c{k}_f{f}': comps[nm][f] for k, nm in enumerate(comps) for f in FOLDS}); print('dumped', len(comps), flush=True); sys.exit(0)
names = list(comps); rows = np.isin(fold, FOLDS); GRID = np.round(np.arange(0, 1.0001, 0.125), 3)
def corr(w):
    c = {f: np.zeros((N_, 3)) for f in range(5)}
    for f in FOLDS:
        for k, nm in enumerate(names): c[f][:, 1] += w[k] * comps[nm][f]
    return c
APE_REF = E.ape(BP.REF_BEST, y)[:, 1]
def cv_rows(w, mask): return float(np.nanmean(E.ape(BP.pipe(BP.add(BP.BEST, corr(w))), y)[mask, 1]))
def fit(mask):
    w = np.zeros(len(names)); cur = cv_rows(w, mask)
    for _ in range(2):
        for k in range(len(names)):
            best = (cur, w[k])
            for g in GRID:
                if g == w[k]: continue
                t = w.copy(); t[k] = g; s = cv_rows(t, mask)
                if s < best[0] - 1e-6: best = (s, g)
            w[k] = best[1]; cur = best[0]
    return w
P = np.full_like(BP.H, np.nan); W = {}
for f in FOLDS:
    m_train = rows & (fold != f); w = fit(m_train); W[f] = w.tolist(); P[fold == f] = BP.pipe(BP.add(BP.BEST, corr(w)))[fold == f]
a = E.ape(P, y)[:, 1]
print('weights per outer fold (chosen on the other folds):', json.dumps({f: dict(zip(names, W[f])) for f in FOLDS}), flush=True)
print('NESTED STACK CV on folds %s: %.4f vs best %.4f  d %.4f | per fold %s' % (''.join(map(str, FOLDS)), np.nanmean(a[rows]), np.nanmean(APE_REF[rows]), np.nanmean(a[rows]) - np.nanmean(APE_REF[rows]), {f: round(float(np.nanmean(a[fold == f]) - np.nanmean(APE_REF[fold == f])), 4) for f in FOLDS}), flush=True)
W_ALL = fit(rows).tolist(); print('deploy weights (fit on all reported folds, for the full-data model):', json.dumps(dict(zip(names, W_ALL))), flush=True)
if SAVE:
    np.savez_compressed(X / f'lane_cv_20261108/out/{SAVE}.npz', P=P, names=np.array(names), w_all=np.array(W_ALL), **{f'w{f}': np.array(W[f]) for f in FOLDS}); print('saved', SAVE)
    if os.environ.get('SAVE_COMPS'):  # component vectors per outer fold (CV column, all rows), for cv_apply_20261110.py check
        np.savez_compressed(X / f'lane_cv_20261108/out/{SAVE}_comps.npz', names=np.array(names), **{f'c{k}_f{f}': comps[nm][f] for k, nm in enumerate(names) for f in FOLDS}); print('saved comps', SAVE)
