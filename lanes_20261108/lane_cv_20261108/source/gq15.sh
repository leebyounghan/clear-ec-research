#!/bin/bash
# CV lane: node threshold 0.25 on the shipped single-view pre-only peaks (control for the TTA t0.25 gain), folds 0 and 4
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 PREP_WORKERS=2
PY=/workspace/venv/bin/python
F=/workspace/clear-ec-research/experiments/green_track_20261028/external_srep
date; $PY source/graph_prep_sp_20261108.py raw_s46_t025.npz "" 3 380 300 0.25 0
date; $PY source/graph_prep_sp_20261108.py raw_s47_t025.npz $F/s47_preonly 3 380 300 0.25 0
date; $PY source/graph_prep_sp_20261108.py raw_s48_t025.npz $F/s48_preonly 3 380 300 0.25 0
until grep -q QDONE logs/gq14.log; do sleep 60; done
for g in raw_s46_t025 raw_s47_t025 raw_s48_t025; do date; $PY source/graph_train_cv_20261108.py $g --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/$g.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 04; done
date; echo QDONE
