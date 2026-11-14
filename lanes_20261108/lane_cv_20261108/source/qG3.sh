#!/bin/bash
# CV lane: order-loss graph members for the s46/s47/s48 sets (folds 0,4 screen, then 1,2,3, merge, full data) on GPU 2 beside qR3.sh, after the window member (qW.sh).
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/workspace/venv/bin/python; G=/workspace/clear-ec-research/experiments/lane_cv_20261108/graphs
ARGS="--order 1.0 --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx"
until grep -q QWDONE logs/qW.log; do sleep 120; done
echo "$(date -u) start"
for s in s46 s47 s48; do [ -f graph_runs/gord_${s}_tta4_04.npz ] || $PY source/graph_order_cv_20261110.py gord_${s}_tta4_04 --graphs $G/tta_${s}_tta4.npz $ARGS --folds 04; done
echo "$(date -u) SCREEN_DONE"
for s in s46 s47 s48; do [ -f graph_runs/gord_${s}_tta4_f123.npz ] || $PY source/graph_order_cv_20261110.py gord_${s}_tta4_f123 --graphs $G/tta_${s}_tta4.npz $ARGS --folds 123; done
$PY -c "
import numpy as np
for s in ['s46', 's47', 's48']:
    a = np.load(f'graph_runs/gord_{s}_tta4_04.npz'); b = np.load(f'graph_runs/gord_{s}_tta4_f123.npz')
    np.savez_compressed(f'graph_runs/gord_{s}_tta4_5f.npz', **{f'f{f}': (a if f in (0, 4) else b)[f'f{f}'] for f in range(5)}); print('merged', s)"
echo "$(date -u) FIVE_DONE"
for s in s46 s47 s48; do $PY source/graph_order_cv_20261110.py gord_${s}_tta4_full --graphs $G/tta_${s}_tta4.npz $ARGS --folds full; done
echo "$(date -u) QG3DONE"
