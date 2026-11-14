import sys; sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L, numpy as np
BP = L.BP; O = BP.O / 'graph_sweep'; N = L.N
def g(names): R = [np.load(O / f'{q}.npz') for q in names]; return {f: np.mean([z[f'f{f}'][:, 2] for z in R], 0) for f in range(5)}
old = {f: BP.GRAPH[f][:, 2] for f in range(5)}
for names in (['d64_l5_noctx_s10', 'd64_l5_pre47_s10', 'd64_l5_pre48_s10'], ['d64_l5_noctx_s10'], ['d64_l5_noctx_s10', 'd64_l5_pre47_s10', 'd64_l5_pre48_s10', 'd64_l5_pre47_s3', 'd64_l5_pre48_s3'], ['d64_l5_pre47_s10', 'd64_l5_pre48_s10']):
    new = g(names); L.evaluate('graph HEX ens ' + '+'.join(n.replace('d64_l5_', '') for n in names), {f: new[f] - old[f] for f in range(5)})
# graph HEX weight scaling
for w in (0.8, 1.2, 1.4):
    L.evaluate(f'graph HEX scaled x{w}', {f: (w - 1) * old[f] for f in range(5)})
