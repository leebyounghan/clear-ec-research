"""ViT-L arm read (D-SYN-002): runs/synthft_vitl_f{f}_s46 against the plain ViT-L s46 (runs/vitl16_avg_f{f}_s46, same SSL and recipe) and the candidate; copy of read_arm_vs_plain_20261111.py with ViT-L paths.
Original: Arm read against plain DINOv3-B s46 and the candidate (generalised from read_distill_20261111.py; --arm NAME reads runs/NAME_f{f}_dinov3b16_s46).
Original: Distillation student read (coordinator 11-11). Folds 0+4: student (bb_runner, tracks raw/ema998/ema999/swa6) vs (a) plain DINOv3-B s46
(lane_gpu3_20261106, family runner) and (b) the integrated candidate (integration_rehearsal_20261110/out/integrated_oof.npz P_main) on
the same validation rows. Member error per row = mean over epochs 8-12 of the per-epoch APE (as read_bb); candidate = its single OOF
prediction. Gap closed = (a - student) / (a - b) on the mean of the three metrics (two-fold mean of per-fold scores).
Caveat: teacher values on the student's training rows come from models that saw the held-out fold's labels (slightly optimistic).
Usage: read_distill_20261111.py [--window 8,12] [--folds 0,4]"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
X = Path('/workspace/clear-ec-research/experiments'); K = ['CD', 'CV', 'HEX']
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ARM = arg('--arm', 'distill'); lo, hi = [int(v) for v in arg('--window', '8,12').split(',')]; folds = [int(v) for v in arg('--folds', '0,4').split(',')]
def member(run, lo, hi):
    P = []
    for e in range(lo, hi + 1):
        f = run / f'predictions_epoch_{e:02d}.npz'
        if not f.exists():
            return None
        with np.load(f, allow_pickle=True) as z:
            P.append((z['indices'].astype(int), z['pred'].astype(float), z['target'].astype(float)))
    ix = P[0][0]; y = P[0][2]
    return ix, np.nanmean([E.ape(p[1], y) for p in P], 0), y
z = np.load(X / 'integration_rehearsal_20261110/out/integrated_oof.npz', allow_pickle=True)
cand = dict(zip(z['idx'].astype(int), z['P_main']))
res = {'window': [lo, hi], 'folds': folds, 'caveat': 'teacher OOF on training rows saw the held-out fold labels: slightly optimistic'}
arms = {'a_plain_dinov3b_s46': lambda f: X / (f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46/raw/runs/fold{f}/dinov3_l16t' if f in (0, 4) else f'lane_cd_20261108/arms/vitl16_f{f}_s46/raw/runs/fold{f}/dinov3_l16t')}  # key name kept; this is the plain ViT-L s46 (CD lane runs for folds 1-3)
for t in ('raw', 'swa6'):
    arms[f'student_{t}'] = (lambda t: lambda f: X / f'lane_backbone_20261108/runs/synthft_vitl_f{f}_s46/{t}/runs/fold{f}/dinov3_l16t')(t)
per = {}
for name, path in arms.items():
    per[name] = {}
    for f in folds:
        m = member(path(f), lo, hi)
        if m is None:
            continue
        ix, err, y = m; s = E.official(err); per[name][f] = {k: round(s[k], 4) for k in K + ['mean']}
        if name == 'a_plain_dinov3b_s46':
            pc = np.array([cand[i] for i in ix]); sc = E.official(E.ape(pc, y)); per.setdefault('b_candidate', {})[f] = {k: round(sc[k], 4) for k in K + ['mean']}
for name, d in per.items():
    if len(d) == len(folds):
        d['two_fold'] = {k: round(float(np.mean([d[f][k] for f in folds])), 4) for k in K + ['mean']}
res['scores'] = per
a, b = per['a_plain_dinov3b_s46'].get('two_fold'), per['b_candidate'].get('two_fold')
for t in ('raw', 'ema998', 'ema999', 'swa6'):
    s = per.get(f'student_{t}', {}).get('two_fold')
    if s and a and b and len(per[f'student_{t}']) > 1:
        res[f'gap_closed_{t}'] = {k: round((a[k] - s[k]) / (a[k] - b[k]), 3) if a[k] != b[k] else None for k in K + ['mean']}
print(json.dumps(res, indent=1))
