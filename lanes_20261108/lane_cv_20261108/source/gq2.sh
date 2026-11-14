#!/bin/bash
# GPU 2 queue 2 (CV lane): graph network on more / larger windows, folds 0 and 4, 3 seeds
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
until grep -q QDONE logs/gq1.log && grep -q ALLDONE logs/prep_batch2.log; do sleep 30; done
for g in k5_t035 k5_t035_p47 k5_t035_p48 box2_t035 box2_t035_p47 box2_t035_p48; do
  date; $PY source/graph_train_cv_20261108.py $g --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/$g.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 04
done
date; echo QDONE
