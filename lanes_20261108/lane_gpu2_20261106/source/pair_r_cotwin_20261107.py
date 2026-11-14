"""Within-pair read of within-donor training, fold 0, secondary and descriptive in the gate. Donors with exactly two validation rows; r between the difference of log labels and of log predictions, mean of epochs 16-20, arm against reference, with a pair bootstrap on the change in r."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
X = Path('/workspace/clear-ec-research/experiments')
b = E.load_base(); gp = E.donors(b['ids']); pos = {int(i): n for n, i in enumerate(b['idx'])}


def last5(root, f=0):
    run = X / root / 'runs' / f'fold{f}' / 'cpdino_vitb'; out = []
    for e in range(16, 21):
        with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64), z['target'].astype(np.float64)))
    return out[0][0], out[0][2], np.mean([np.log(p) for _, p, _ in out], 0)


ix, y, a = last5('lane_gpu2_20261106/runs/cotwin_mb4_f0_s46'); ix0, y0, r = last5('phase2_pooling_replication_20261026/f0_s46_std')
assert np.array_equal(ix, ix0) and np.array_equal(y, y0)
g = gp[np.array([pos[int(i)] for i in ix])]; pairs = []
for d in np.unique(g):
    w = np.where(g == d)[0]
    if len(w) == 2: pairs.append(w)
pairs = np.array(pairs); ly = np.log(y.clip(1e-9)); ok = (y[pairs[:, 0]] > 0).all(1) & (y[pairs[:, 1]] > 0).all(1); pairs = pairs[ok]
dy = ly[pairs[:, 0]] - ly[pairs[:, 1]]; da = a[pairs[:, 0]] - a[pairs[:, 1]]; dr = r[pairs[:, 0]] - r[pairs[:, 1]]; rng = np.random.default_rng(0); out = dict(n_pairs=len(pairs))
for j, k in enumerate(['CD', 'CV', 'HEX']):
    ra, rr = np.corrcoef(dy[:, j], da[:, j])[0, 1], np.corrcoef(dy[:, j], dr[:, j])[0, 1]; bs = []
    for _ in range(2000):
        s = rng.integers(0, len(pairs), len(pairs)); bs.append(np.corrcoef(dy[s, j], da[s, j])[0, 1] - np.corrcoef(dy[s, j], dr[s, j])[0, 1])
    out[k] = dict(r_arm=round(float(ra), 3), r_ref=round(float(rr), 3), change=round(float(ra - rr), 3), ci95=[round(float(v), 3) for v in np.percentile(bs, [2.5, 97.5])])
print(json.dumps(out))
Path(X / 'lane_gpu2_20261106/reads/cotwin_f0_within_pair.json').write_text(json.dumps(out, indent=1))
