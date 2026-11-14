#!/bin/bash
# GPU 2 queue 3 (CV lane): graph network on graphs from the fine-tuned sg2 specialists, folds 0 and 4, 3 seeds
cd /workspace/clear-ec-research/experiments/lane_cv_20261108
export CUDA_VISIBLE_DEVICES=2
PY=/workspace/venv/bin/python
until grep -q QDONE logs/gq2.log && grep -q ALLDONE logs/prep_batch3.log; do sleep 30; done
for g in ft_s46_t035 ft_s47_t035 ft_s48_t035 ft_s49_t035 ft_s50_t035; do
  date; $PY source/graph_train_cv_20261108.py $g --graphs /workspace/clear-ec-research/experiments/lane_cv_20261108/graphs/$g.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 3 --no-ctx --folds 04
done
date; echo QDONE
