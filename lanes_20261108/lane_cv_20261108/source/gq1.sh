#!/bin/bash
# GPU 2 queue 1 (CV lane): graph network on spacing-prior graphs, folds 0 and 4, 3 seeds, shipped d64_l5 settings
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
until grep -q ALLDONE logs/prep_batch1.log; do sleep 20; done
for g in sp07_t02 sp07_t02_p47 sp07_t02_p48 sp06_t05f07 sp06_t05f07_p47 sp06_t05f07_p48; do
  date; $PY source/graph_train_cv_20261108.py $g --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/$g.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 04
done
date; echo QDONE
