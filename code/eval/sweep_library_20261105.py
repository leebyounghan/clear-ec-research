"""Member library for the combination sweep (2026-11-05). Assembles every atomic prediction family into a row x member x metric array, from inventory_files.json. Complete = all five folds, each row from its own fold's validation. Blends are left out,
re-inferences (bf16, fp16) and near copies are dropped (log correlation > 0.9995 on all metrics, the better one stays). Recipe A seeds come from load_members. Families split across two experiment folders (anchor_quality seeds) are joined. Writes sweep/library.npz: names, group, preds, complete mask."""
import json, re, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
E = C.E; K = list(C.K)
ROOT = C.X; OUT = ROOT / 'meta_ceiling_20261103/sweep'
b = E.load_base(); y, fold = b['y'], b['fold']; pos = {int(i): n for n, i in enumerate(b['idx'])}; N = len(y)
inv = json.load(open(OUT / 'inventory_files.json'))
BLEND = {'B_hybrid_w0p5', 'C_hybrid_lofo_w', 'A_recipeA_tta_scalar', 'historical_five_calibrated', 'blend', 'ensemble', 'current4', 'current5', 'ensemble_bf16', 'ensemble_fp16', 'standalone', 'single', 'tta4', 'seed42', 'metric35_bf16', 'metric35_fp16', 'predbag2_35_bf16', 'predbag2_35_fp16', 'seed42_bf16', 'seed42_fp16', 'seed77_bf16', 'seed77_fp16', 'seed2026_bf16', 'seed2026_fp16', 'anchor_quality_bf16', 'anchor_quality_fp16', 'full_quality_bf16', 'full_quality_fp16'}
SKIP = re.compile(r'roi_crop|recipeA_analysis|hybrid_20260905/predictions|five_fold_predictions|fulldata_recipeA|final_candidate|deploy_review|inner\d|nested_oof|regression_std_s4[67]|f\{f\}_s4[67]_std|arms/std/')
fam = {}
for r in inv:
    for a in r.get('arrays', []):
        if a['key'] in BLEND or a['key'] == 'pred_single' or SKIP.search(r['family']):
            continue
        if a['key'] == 'pred' and (ROOT / r['path']).name == 'predictions.npz' and (ROOT / r['path']).with_name('predictions_tta4.npz').exists():
            continue                                                    # the TTA version stands in for it
        fam.setdefault((r['family'], a['key']), []).append(r['path'])
def label(f, k):
    s = f.replace('/runs/fold{f}/cpdino_vitb/predictions_tta4.npz', '').replace('/runs/fold{f}/cpdino_vitb/predictions.npz', '').replace('/best_oof.npz', '').replace('_seed42_c256k1_mean_multitask_slidesmooth', '').replace('_multitask_slidesmooth', '')
    s = s.replace('{f}', '').replace('/runs', '').replace('fold/', '').replace('/fold', '')
    return s + ('' if k in ('pred', 'pred_tta4') else ':' + k)
P, names = {}, {}
for (f, k), paths in fam.items():
    a = np.full((N, 3), np.nan)
    for p in paths:
        with np.load(ROOT / p, allow_pickle=True) as z:
            ik = next(q for q in ('indices', 'idx', 'rows', 'index') if q in z.files); r = np.array([pos.get(int(i), -1) for i in z[ik]]); ok = r >= 0; a[r[ok]] = z[k][ok].astype(np.float64)
    s = np.isfinite(a).all(1) & (y > 0).all(1)
    if s.sum() < 500: continue
    if max(abs(np.log(np.median(a[s] / y[s], 0)))) > 0.5: a = a / np.median(a[s] / y[s], 0)
    P[label(f, k)] = a
# join families split by experiment folder
for sd in ('77', '2026'):
    a = np.full((N, 3), np.nan); parts = [k for k in P if re.search(rf'anchor_quality_f?_seed{sd}_', k)]
    for k in parts: a = np.where(np.isfinite(a), a, P[k])
    if np.isfinite(a).all(): P[f'anchor_quality_s{sd}'] = a; [P.pop(k) for k in parts]
seeds, single, hist = C.load_members(b)
for s, a in seeds.items(): P[f'recipeA_s{s}_tta'] = a
P = {k: v for k, v in P.items()}; print('families', len(P))
def score(a):
    s = np.isfinite(a).all(1); return E.official(E.ape(a[s], y[s]))['mean'], int(s.sum())
names = sorted(P, key=lambda k: (-score(P[k])[1], score(P[k])[0])); keep = []
for n in names:
    a = P[n]; dup = False
    for m in keep:
        c = P[m]; s = np.isfinite(a).all(1) & np.isfinite(c).all(1)
        if s.sum() > 400 and min(np.corrcoef(np.log(a[s, j]), np.log(c[s, j]))[0, 1] for j in range(3)) > 0.9995: dup = True; break
    if not dup: keep.append(n)
print('after copies removed', len(keep))
def grp(n):
    if re.search(r'recipeA_s|regression_std|g3_hex6aux|local_pool|pooling|backbone|bridge2|ssl_seed|masked|e3_box|label_clean|formula_head|invariance|zoom|cotwin|lejepa|style_spec|crossfold|fullnative|foundation', n): return 'whole_frame'
    return 'crop'
arr = np.stack([P[n] for n in keep], 1); comp = np.isfinite(arr).all((0, 2)); g = np.array([grp(n) for n in keep])
np.savez(OUT / 'library.npz', names=np.array(keep), group=g, preds=arr.astype(np.float32), complete=comp)
for i, n in enumerate(keep):
    sc, ct = score(arr[:, i]); s0 = E.official(E.ape(b['B'][np.isfinite(arr[:, i]).all(1)], y[np.isfinite(arr[:, i]).all(1)]))['mean']; print(('C' if comp[i] else 'p'), g[i][:5], round(sc, 3), round(sc - s0, 3), ct, sorted(np.unique(fold[np.isfinite(arr[:, i]).all(1)]).tolist()), n[:90])
print('complete', int(comp.sum()), 'partial', int((~comp).sum()))
