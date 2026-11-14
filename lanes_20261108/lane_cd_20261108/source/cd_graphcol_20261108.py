"""Do the graph / simulation-embedding corrections carry CD signal? Their CD column is zeroed in the best pipeline."""
import numpy as np
from cd_common import *
O = BP.O
ALL = yc > 0
for q in ('d64_l5_noctx_s10', 'd64_l5_pre47_s3', 'd64_l5_pre48_s3'):
    z = np.load(O / 'graph_sweep' / f'{q}.npz'); print(q, z.files[:8], flush=True)
_R = [np.load(O / 'graph_sweep' / f'{q}.npz') for q in ('d64_l5_noctx_s10', 'd64_l5_pre47_s3', 'd64_l5_pre48_s3')]
G = {f: np.mean([z[f'f{f}'] for z in _R], 0).astype(np.float64) for f in range(5)}
S_ = np.load(O / 'sim_pred' / 'sim2_l8_ongraph_nested.npz'); SIM = {f: S_[f'f{f}'].astype(np.float64) for f in range(5)}
for nm, Cc in (('graph', G), ('sim', SIM)):
    for f in range(5):
        print(nm, f, 'CD col sd', round(float(np.std(Cc[f][:, 0])), 4), 'nonzero', int((Cc[f][:, 0] != 0).sum()), flush=True)
Hc = BP.H[:, 0]
for nm, Cc in (('graph', G), ('sim', SIM)):
    for w in (0.25, 0.5, 1.0):
        cor = np.zeros_like(Hc)
        for f in range(5): cor[fold == f] = Cc[f][fold == f, 0]
        p = stretch(Hc * np.exp(w * cor)); rep(f'{nm} CD col w={w}', p, rows=ALL)
