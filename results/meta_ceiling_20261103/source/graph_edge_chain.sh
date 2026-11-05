#!/bin/bash
# Builds the edge-evidence graphs for the three pre-only detectors, each only after the previous file exists, then writes the jobs for the GPU 3 queue.
cd /workspace/clear-ec-research; PY=/workspace/venv/bin/python; A=experiments/meta_ceiling_20261103; G=experiments/green_track_20261028; S=$A/source/cell_graph_prep2_20261105.py; O=$A/sweep
export CUDA_VISIBLE_DEVICES=
$PY $S cell_graphs_edge46.npz '' 3 && [ -f $O/cell_graphs_edge46.npz ] || exit 1
$PY - <<PYEOF
import json, os
A='/workspace/clear-ec-research/$A'; P='$PY'; S='$A/source/cell_graph_sweep2_20261105.py'
J=lambda n,*a: dict(name=n, done=A+'/sweep/graph_sweep/'+n+'.json', cmd=[P,S,n,'--no-ctx','--layers','5','--seeds','3',*a])
g=['--graphs','cell_graphs_edge46.npz']
jobs=[dict(name='tok8_l5_pre46_s3', done=A+'/sweep/graph_sweep/tok8_l5_pre46_s3.json', cmd=[P,'$A/source/cell_graph_sweep_tokens_20261105.py','tok8_l5_pre46_s3','--no-ctx','--layers','5','--seeds','3','--graphs','cell_graphs_xy.npz','--tokens','node_tokens.npz','--token-dims','8']),
 J('edge_geom_46',*g,'--edge-cols','3,4'), J('edge_ridge_46',*g,'--edge-cols','0,1,2'), J('edge_all_46',*g,'--edge','5'), J('edge_all_node_46',*g,'--edge','5','--node-extra'), J('node_only_46',*g,'--node-extra')]
p=A+'/sweep/jobs_graph_c.json'; json.dump(jobs, open(p+'.tmp','w'), indent=1); os.replace(p+'.tmp',p); print('jobs written')
PYEOF
$PY $S cell_graphs_edge47.npz $G/external_srep/s47_preonly 3
$PY $S cell_graphs_edge48.npz $G/external_srep/s48_preonly 3
echo chain finished
