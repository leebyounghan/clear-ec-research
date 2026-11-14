#!/bin/bash
# CV lane: graphs + graph network (shipped d64_l5 settings, 5 folds, 3 seeds) for pre-only detectors seeds 49 and 50
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2 PREP_WORKERS=2
PY=/workspace/venv/bin/python
until grep -q TTADONE logs/preonly_more.log; do sleep 30; done
for s in s49 s50; do date; $PY source/graph_prep_sp_20261108.py pre_$s.npz out/preonly_more/$s 3 380 300 0.35 0; done
for s in s49 s50; do date; $PY source/graph_train_cv_20261108.py pre_$s --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/pre_$s.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 01234; done
date; echo QDONE
