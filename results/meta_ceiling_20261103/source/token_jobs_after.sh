#!/bin/bash
# When the node tokens exist, writes the job list that the GPU 3 graph queue is watching.
A=/workspace/clear-ec-research/experiments/meta_ceiling_20261103; X=experiments/meta_ceiling_20261103; PY=/workspace/venv/bin/python
until [ "$(tail -1 $A/sweep/node_tokens.log | cut -c1-4)" = "done" ]; do grep -q Traceback $A/sweep/node_tokens.log && exit 1; sleep 60; done
$PY - <<PYEOF
import json, os
P='$PY'; S='$X/source/cell_graph_sweep_tokens_20261105.py'
J=lambda n,*a: dict(name=n, done='$A/sweep/graph_sweep/'+n+'.json', cmd=[P,S,n,'--no-ctx',*a])
jobs=[J('tok32_l5_pre46_s3','--layers','5','--seeds','3','--graphs','cell_graphs_xy.npz','--tokens','node_tokens.npz'),
      J('tok32_l5_pre47_s3','--layers','5','--seeds','3','--graphs','cell_graphs_pre47_xy.npz','--tokens','node_tokens_pre47.npz'),
      J('tok8_l5_pre46_s3','--layers','5','--seeds','3','--graphs','cell_graphs_xy.npz','--tokens','node_tokens.npz','--token-dims','8'),
      J('tok32_l5_d128_pre46_s3','--layers','5','--d','128','--seeds','3','--graphs','cell_graphs_xy.npz','--tokens','node_tokens.npz'),
      J('tok32_l3_pre46_s3','--layers','3','--seeds','3','--graphs','cell_graphs_xy.npz','--tokens','node_tokens.npz')]
p='$A/sweep/jobs_graph_c.json'; json.dump(jobs, open(p+'.tmp','w'), indent=1); os.replace(p+'.tmp',p)
PYEOF
