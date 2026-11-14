"""Read of the node-token arms against the geometry-only reference on the same graphs (lane GPU 1, 2026-11-07). Corrections on CV and HEX only, as in the best pipeline. Gate in protocol.json."""
import json, sys
import numpy as np
sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
G = BP.O / 'graph_sweep'
def cor(n):
    z = np.load(G / f'{n}.npz'); return {f: z[f'f{f}'].astype(np.float64) * BP._M for f in range(5)}
REF = BP.pipe(cor('d64_l5_pre48_s3'))
BP.report('reference against hex6 + stretch', REF, BP.REF_HEX6)
out = []
for n in sys.argv[1:]:
    if not (G / f'{n}.npz').exists():
        print('missing', n, flush=True); continue
    out.append(BP.report(n, BP.pipe(cor(n)), REF))
open('experiments/lane_gpu1_20261106/reads/node_tokens.json', 'w').write(json.dumps(out, indent=1))
