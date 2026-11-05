#!/bin/bash
# Rebuilds the two graph files with node coordinates, then samples SSL tokens at the nodes on GPU 3.
cd /workspace/clear-ec-research; PY=/workspace/venv/bin/python; A=experiments/meta_ceiling_20261103; G=experiments/green_track_20261028; S=$A/source/cell_graph_prep_20261105.py
CUDA_VISIBLE_DEVICES= $PY $S cell_graphs_xy.npz '' 3
CUDA_VISIBLE_DEVICES= $PY $S cell_graphs_pre47_xy.npz $G/external_srep/s47_preonly 3
CUDA_VISIBLE_DEVICES=3 $PY $A/source/node_tokens_20261105.py cell_graphs_xy.npz node_tokens.npz cell_graphs_pre47_xy.npz node_tokens_pre47.npz
