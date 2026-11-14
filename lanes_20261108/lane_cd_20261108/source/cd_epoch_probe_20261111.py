"""D-CDTRAIN-001 part 1 (CPU probe): CD MAPE per epoch on five folds for recipe A s46/s47 (single view e1..e20 where stored, TTA4 at e20) and
ViT-L s46/s47 (single view e1..e12, final TTA), single epoch and running snapshot mean (exp mean log over e8..e). Raw member MAPE and, to separate
level drift from ranking, MAPE after a per-fold MAPE-optimal scalar fit on the other four folds (nested). Question: is CD still falling at the end?"""
import os, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
from cd_common import *
ok = yc > 0


def vec(p):
    z = np.load(p); v = np.full(len(yc), np.nan); key = 'pred' if 'pred' in z.files else 'pred_tta4'; v[[R.pos[int(i)] for i in z['indices']]] = z[key][:, 0]; return v


def mape(q): m = ok & np.isfinite(q); return float(np.mean(np.abs(q[m] - yc[m]) / yc[m]) * 100)


def scaled(q):
    out = np.full(len(yc), np.nan)
    for f in range(5):
        tr = (fold != f) & ok & np.isfinite(q); s = C.wmed(yc[tr] / q[tr], q[tr] / yc[tr]); out[fold == f] = q[fold == f] * s
    return mape(out)


def member_dirs(kind, seed):
    if kind == 'A':
        return {f: X / root / 'runs' / f'fold{f}' / 'cpdino_vitb' for f, root in C.RECIPE[seed].items()}
    return {f: ((X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46') if (f in (0, 4) and seed == 46) else (X / f'lane_cd_20261108/arms/vitl16_f{f}_s{seed}')) / f'raw/runs/fold{f}/dinov3_l16t' for f in range(5)}


res = {}
for kind, seeds, last in (('A', (46, 47), 20), ('V', (46, 47), 12)):
    for s in seeds:
        dirs = member_dirs(kind, s); rows = []; logs = {}
        for e in range(1, last + 1):
            q = np.full(len(yc), np.nan); have = True
            for f, d in dirs.items():
                pth = d / f'predictions_epoch_{e:02d}.npz'
                if not pth.exists(): have = False; break
                v = vec(pth); q[fold == f] = v[fold == f]
            if not have: continue
            logs[e] = np.log(q)
            snap = np.exp(np.mean([logs[k] for k in logs if k >= 8], 0)) if e >= 8 else None
            rows.append(dict(epoch=e, single=round(mape(q), 4), single_scaled=round(scaled(q), 4),
                             snap_from8=(round(mape(snap), 4) if snap is not None else None), snap_from8_scaled=(round(scaled(snap), 4) if snap is not None else None)))
        tta = np.full(len(yc), np.nan); fin = 'predictions_tta4.npz' if kind == 'A' else 'predictions_final_tta.npz'
        okf = all((d / fin).exists() for d in dirs.values())
        if okf:
            for f, d in dirs.items():
                v = vec(d / fin); tta[fold == f] = v[fold == f]
        res[f'{kind}_s{s}'] = dict(per_epoch=rows, final_tta=(round(mape(tta), 4) if okf else None), final_tta_scaled=(round(scaled(tta), 4) if okf else None))
        print(f'{kind} s{s}', json.dumps(rows[-6:]), 'final TTA', res[f"{kind}_s{s}"]['final_tta'], flush=True)
json.dump(res, open(X / 'lane_cd_20261108/reads/cd_epoch_probe_20261111.json', 'w'), indent=1); print('saved', flush=True)
