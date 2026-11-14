#!/bin/bash
# CV lane: TTA t0.25 graph members for pre-only detectors seeds 49/50, five folds directly
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 PREP_WORKERS=2
PY=/workspace/venv/bin/python
until grep -q TTADONE logs/tta_more.log; do sleep 60; done
for s in s49 s50; do date; $PY source/graph_prep_sp_20261108.py tta_${s}_t025.npz out/tta_peaks/${s}_tta4 3 380 300 0.25 0; done
until grep -q QDONE logs/gq16.log; do sleep 60; done
for s in s49 s50; do date; $PY source/graph_train_cv_20261108.py tta_${s}_t025_5f --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/tta_${s}_t025.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 01234; done
date; echo QDONE
