#!/bin/bash
# Builds the extra cell-graph sets, one after the other.
cd /workspace/clear-ec-research; PY=/workspace/venv/bin/python; A=experiments/meta_ceiling_20261103; G=experiments/green_track_20261028; S=$A/source/cell_graph_prep_20261105.py
export CUDA_VISIBLE_DEVICES=
$PY $S cell_graphs_k1.npz '' 1
$PY $S cell_graphs_k5.npz '' 5
$PY $S cell_graphs_whole.npz '' 0
$PY $S cell_graphs_pre47.npz $G/external_srep/s47_preonly 3
$PY $S cell_graphs_pre48.npz $G/external_srep/s48_preonly 3
$PY $S cell_graphs_sg2.npz $G/sg2_specialist 3
echo finished
