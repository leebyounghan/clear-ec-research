#!/bin/bash
# Lane GPU 3 preflight: runner parity on recipe A, then speed and memory of the first two families.
cd /workspace/clear-ec-research
export CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
L=experiments/lane_gpu3_20261106; PY=/workspace/venv/bin/python; RUN=$L/source/family_recipe_20261106.py
REF=experiments/phase2_pooling_replication_20261026/f0_s46_std
SSL=experiments/phase2_crossfold_20261015/fold0/ssl5/run/last.pt
echo "start parity $(date -u +%FT%TZ)"
$PY $RUN parity --family cpdino_b8 --fold 0 --seed 46 --parity-ref $REF; echo "rc parity $?"
$PY $RUN smoke --family cpdino_b8 --fold 0 --seed 46 --root $L/runs/_smoke --ssl-epochs 0 --ssl-from $SSL --updates 100; echo "rc gradnorm $?"
$PY $RUN smoke --family dinov3_b16 --fold 0 --seed 46 --root $L/runs/_smoke --updates 25; echo "rc dinov3_b16 $?"
$PY $RUN smoke --family dinov3_b16_s8 --fold 0 --seed 46 --root $L/runs/_smoke --updates 25; echo "rc dinov3_b16_s8 $?"
echo "done $(date -u +%FT%TZ)"
