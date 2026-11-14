#!/bin/bash
# Seed audit, detector-seed check of gnn:hex_pre46_s5 (coordinator 2026-11-11): the same HEX-only graph net (5 training seeds) on the
# pre47 detector's cell graphs, five folds, on GPU 2 after the cvw3 s47 fold-4 run (wrapper pid given) has finished.
# usage: gnn_pre47_after.sh WAIT_PID
cd /workspace/clear-ec-research/experiments/lane_hex_20261108; Q=logs/seed_audit_gpu2.log
while kill -0 $1 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) start gnn hex_pre47_s5 gpu 2" >> $Q; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 /workspace/venv/bin/python source/gnn_hex.py hex_pre47_s5 --graphs /workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/cell_graphs_pre47.npz --lw 0,0,1 --seeds 5 --threads 2 > logs/hex_pre47_s5.log 2>&1
echo "$(date -u +%FT%TZ) end gnn hex_pre47_s5 rc $? seconds $(( $(date +%s) - t0 ))" >> $Q
