"""True-form table (coordinator 03:33 EST 11-14): fold 0, members as epochs 16-20 four-flip TTA (tta/<name>_tta4_e16_20.npz), each arm against
control P in the same form, donor-clustered paired bootstrap (2,000). Arms: whatever files exist among DINOv3-B encoder LR 2e-5, CD-weighted
loss, box encoder LR 3e-5, wd 0.05, appear 0.5; yardstick: the incumbent plain s46 fold-0 member (family runner, same form) against P.
Writes reads/truetta_table_20261114.json."""
import json, numpy as np
from cd_common import *
import cd_read_20261108 as R
y3 = BP.y.astype(float); f0 = fold == 0
def vec3(p):
    z = np.load(p); v = np.full((len(yc), 3), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; return v
T = LANE / 'tta'; P = vec3(T / 'screen_plain_f0_tta4_e16_20.npz')
ARMS = {'enc_lr_2e-5': 'sweep_elr2e5_f0', 'cd_weighted_loss_1.8_0.6_0.6': 'sweep_cdw3_f0', 'box_enc_lr_3e-5 (RTX 4090, micro 8x2)': 'sweep_elr3e5_f0',
        'wd_0.05': 'screen_wd005_f0', 'appear_0.5': 'screen_appear05_f0', 'YARDSTICK incumbent plain s46 (family runner)': 'dinov3_s46_f0'}
def m3(p): return [round(float(np.mean(np.abs(p[f0 & (y3[:, j] > 0), j] / y3[f0 & (y3[:, j] > 0), j] - 1)) * 100), 4) for j in range(3)]
def paired(a, b, j):
    r = f0 & (y3[:, j] > 0); d = (np.abs(a[r, j] / y3[r, j] - 1) - np.abs(b[r, j] / y3[r, j] - 1)) * 100; u, inv = np.unique(donors[r], return_inverse=True)
    s = np.bincount(inv, d); n = np.bincount(inv); rng = np.random.default_rng(0); bs = [s[k].sum() / n[k].sum() for k in (rng.integers(0, len(u), len(u)) for _ in range(2000))]
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
out = {'form': 'epochs 16-20 four-flip TTA, fold 0 (1,791 rows)', 'control_P': m3(P)}
for k, n in ARMS.items():
    pth = T / f'{n}_tta4_e16_20.npz'
    if not pth.exists(): out[k] = 'TTA file not yet written'; continue
    v = vec3(pth); out[k] = dict(cd_cv_hex=m3(v), vs_P={kk: paired(v, P, j) for j, kk in enumerate(('CD', 'CV', 'HEX'))})
print(json.dumps(out, indent=1)); json.dump(out, open(LANE / 'reads/truetta_table_20261114.json', 'w'), indent=1)
