"""D-CDTRAIN-001 probe, train columns: per epoch, five-fold mean of the runs' own history.json entries (online train loss, online train CD MAPE with
augmentation, train-probe CD MAPE on 256 training rows in eval mode) next to the validation CD from cd_epoch_probe_20261111.json (single view,
per-fold scalar). Recipe A s46/s47 (phase-2 runner), ViT-L s46/s47 (raw track)."""
import os, json, numpy as np
os.environ['A2SNAP'] = '0'
import cd_read_20261108 as R
from cd_common import *
probe = json.load(open(X / 'lane_cd_20261108/reads/cd_epoch_probe_20261111.json'))
def hist_dirs(kind, s):
    if kind == 'A': return {f: X / root / 'runs' / f'fold{f}' / 'cpdino_vitb' for f, root in C.RECIPE[s].items()}
    return {f: ((X / f'lane_backbone_20261108/runs/vitl16_avg_f{f}_s46') if (f in (0, 4) and s == 46) else (X / f'lane_cd_20261108/arms/vitl16_f{f}_s{s}')) / f'raw/runs/fold{f}/dinov3_l16t' for f in range(5)}
out = {}
for kind, seeds in (('A', (46, 47)), ('V', (46, 47))):
    for s in seeds:
        H = {f: {h['epoch']: h for h in json.load(open(d / 'history.json'))} for f, d in hist_dirs(kind, s).items()}
        val = {r['epoch']: r['single_scaled'] for r in probe[f'{kind}_s{s}']['per_epoch']}
        rows = []
        for e in sorted(set.intersection(*[set(h) for h in H.values()])):
            g = lambda key, sub=None: float(np.mean([(H[f][e][key][sub] if sub else H[f][e][key]) for f in H if key in H[f][e] and H[f][e][key] is not None])) if all(key in H[f][e] and H[f][e][key] is not None for f in H) else None
            rows.append(dict(epoch=e, val_cd=val.get(e), train_loss=g('online_train_loss'), train_cd_online=g('online_train_mape', 'CD'), train_cd_probe=g('train_probe_score', 'CD')))
        out[f'{kind}_s{s}'] = rows
        print(f'{kind} s{s}')
        for r in rows:
            if r['epoch'] >= 6:
                print('  e%02d val %s | train loss %.4f | train CD online %.3f | train CD probe %s' % (r['epoch'], ('%.3f' % r['val_cd']) if r['val_cd'] else '-', r['train_loss'], r['train_cd_online'], ('%.3f' % r['train_cd_probe']) if r['train_cd_probe'] else '-'))
json.dump(out, open(X / 'lane_cd_20261108/reads/cd_epoch_probe_train_20261111.json', 'w'), indent=1)
