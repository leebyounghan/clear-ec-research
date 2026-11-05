import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import stretch_nested_20261103 as N
b = C.E.load_base()
gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True)
prm = N.fit_stretch(b['B'], b['y'], ~gf['seen_leaked'])
print(json.dumps({k: dict(g=float(x[0]), c=float(x[1]), mu=float(x[2])) for k, x in zip(C.K, prm)}))
