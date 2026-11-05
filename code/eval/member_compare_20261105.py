"""Gate read-out for new recipe A members on one fold, seed 46 (2026-11-05). Fold from the environment variable FOLD, default 0. Per arm, at terminal epoch 20 or the latest finished epoch: member vs the matched recipe A member at the same epoch, paired donor-cluster bootstrap; the hybrid with the seed-46 member swapped and with the arm added as a member, one view
(seed 47 and Phase-1 unchanged); error correlation with recipe A; within-pair r on validation fellow eyes. Also the snapshot mean of epochs 11-20 when finished. usage: member_compare_20261105.py name=root ... OUT.json"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
E = C.E; K = list(C.K)
b = E.load_base(); y, fold = b['y'], b['fold']; pos = {int(i): k for k, i in enumerate(b['idx'])}; F_ = int(os.environ.get('FOLD', 0)); rows = np.where(fold == F_)[0]; yy = y[rows]; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]
g = np.unique(E.donors(b['ids'])[rows], return_inverse=True)[1]; sc = np.array(json.load(open(C.X / 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars'][str(F_)]); seeds, single, _ = C.load_members(b)
REF = C.X / C.RECIPE[46][F_]
def pred(root, e):
    with np.load(root / f'runs/fold{F_}/cpdino_vitb/predictions_epoch_{e:02d}.npz') as z:
        p = np.full((len(y), 3), np.nan); p[[pos[int(i)] for i in z['indices']]] = z['pred']
    if 'canonical' in str(root):                                     # predictions are in the zoomed frame: CD x s^2 decodes them
        p[:, 0] = p[:, 0] * CANON[b['idx']] ** 2
    return p[rows]
CANON = np.load(C.X / 'meta_ceiling_20261103/sweep/spacing_t35.npz')['scale'].astype(np.float64)
new = {}
for n, i in zip(rows, [ids[r] for r in rows]):
    m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
    if m: new[(m.group(1), m.group(2), m.group(3).replace('0', 'O'))] = list(rows).index(n) if False else None
rowpos = {int(r): k for k, r in enumerate(rows)}; new = {}
for r in rows:
    m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', ids[r])
    if m: new[(m.group(1), m.group(2), m.group(3).replace('0', 'O'))] = rowpos[int(r)]
P = np.array([(v, new[(k[0], k[1], 'OS')]) for k, v in new.items() if k[2] == 'OD' and (k[0], k[1], 'OS') in new])
def within(p):
    o = {}
    for j, k in enumerate(K):
        s = (yy[P[:, 0], j] > 0) & (yy[P[:, 1], j] > 0); dy = np.log(yy[P[s, 0], j]) - np.log(yy[P[s, 1], j]); dp = np.log(p[P[s, 0], j]) - np.log(p[P[s, 1], j]); o[k] = round(float(np.corrcoef(dy, dp)[0, 1]), 3)
    return o
out = {}; s47 = single[47][rows]; P1 = b['P1'][rows]
for spec in sys.argv[1:-1]:
    name, root = spec.split('=', 1); root = C.X / root; hist = json.load(open(root / f'runs/fold{F_}/cpdino_vitb/history.json')) if (root / f'runs/fold{F_}/cpdino_vitb/history.json').exists() else []
    ep = sorted(int(f.name[-6:-4]) for f in (root / f'runs/fold{F_}/cpdino_vitb').glob('predictions_epoch_*.npz')); e = ep[-1]
    p, base = pred(root, e), pred(REF, e); h0 = 0.5 * (0.5 * (base + s47) * sc) + 0.5 * P1; h1 = 0.5 * (0.5 * (p + s47) * sc) + 0.5 * P1; h2 = 0.5 * (0.5 * (0.5 * (p + base) + s47) * sc) + 0.5 * P1
    d = dict(epoch=e, member={k: round(v, 3) for k, v in C.mape(p, yy).items() if k != 'n'}, ref={k: round(v, 3) for k, v in C.mape(base, yy).items() if k != 'n'})
    for nm, a, c in (('paired_member', p, base), ('paired_hybrid_swap_s46', h1, h0), ('paired_hybrid_add_member', h2, h0)):
        q = E.paired_bootstrap(E.ape(a, yy), E.ape(c, yy), g); d[nm] = [round(q['delta'], 4), round(q['ci95'][0], 4), round(q['ci95'][1], 4)]
    d['err_corr'] = {k: round(float(np.corrcoef(np.log(p[yy[:, j] > 0, j] / yy[yy[:, j] > 0, j]), np.log(base[yy[:, j] > 0, j] / yy[yy[:, j] > 0, j]))[0, 1]), 4) for j, k in enumerate(K)}; d['within_r'] = within(p); d['within_r_ref'] = within(base)
    if e >= 20:
        sn = np.exp(np.mean([np.log(pred(root, q)) for q in range(11, 21)], 0)); sb = np.exp(np.mean([np.log(pred(REF, q)) for q in range(11, 21)], 0)); q = E.paired_bootstrap(E.ape(sn, yy), E.ape(sb, yy), g); d['snapshot_11_20'] = dict(arm=round(C.mape(sn, yy)['mean'], 3), ref=round(C.mape(sb, yy)['mean'], 3), delta=round(q['delta'], 4), ci=[round(v, 4) for v in q['ci95']])
    d['curve_arm'] = [round(C.mape(pred(root, q), yy)['mean'], 3) for q in ep]; d['curve_ref'] = [round(C.mape(pred(REF, q), yy)['mean'], 3) for q in ep]
    out[name] = d; print(name, json.dumps(d), flush=True)
json.dump(out, open(sys.argv[-1], 'w'), indent=1); print('done', flush=True)
