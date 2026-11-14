import sys; sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L, numpy as np
print('rows', L.N, 'idx identity', bool((L.idx == np.arange(L.N)).all()), 'max idx', L.idx.max())
for nm, P in (('sub1', L.BP.REF_SUB1), ('hex6', L.BP.REF_HEX6), ('graph', L.BP.REF_GRAPH), ('best', L.REF)):
    print(nm, 'HEX all5 %.4f  f04 %.4f' % (L.hexape(P, np.ones(L.N, bool)), L.hexape(P, L.S04)), 'per fold', [round(L.hexape(P, L.fold == f), 3) for f in range(5)])
o, _ = L.evaluate('zero correction (sanity)', {f: np.zeros(L.N) for f in range(5)})
