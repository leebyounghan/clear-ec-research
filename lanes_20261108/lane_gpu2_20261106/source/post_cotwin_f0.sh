#!/bin/bash
# Fold 4 of co-twin was cancelled (chain stopped by pid). Fold 0 finishes on its own; this runs its audit and its read.
cd /workspace/clear-ec-research
L=experiments/lane_gpu2_20261106; PY=/workspace/venv/bin/python; W=$L/source/recipeA_cotwin_1gpu_20261106.py; R=$L/runs/cotwin_mb4_f0_s46
A="--root $R --split experiments/ssl_screen_20260911/preflight/fold0.npz --fold 0 --pooling std --seed 46 --ssl-root experiments/phase2_crossfold_20261015/fold0/ssl5 --head-lr 1e-3 --clip 1 --epochs 20 --schedule-epochs 10 --deadline-utc 2026-11-12T00:00:00Z"
while kill -0 59368 2>/dev/null; do sleep 20; done
echo "$(date -u) train f0 ended"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY $W audit $A > $L/logs/cotwin_f0_audit.log 2>&1; echo "$(date -u) audit f0 rc $?"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PY $L/source/read_arm_20261106.py cotwin_f0 'lane_gpu2_20261106/runs/cotwin_mb4_f{f}_s46' 0 --added > $L/logs/read_cotwin_f0.log 2>&1; echo "$(date -u) read f0 rc $?"
echo "$(date -u) post cotwin done"
