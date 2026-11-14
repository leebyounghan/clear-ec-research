#!/bin/bash
# Baseline encoders for the frozen read (lane GPU 1, 2026-11-06). Sequential on GPU 1.
cd /workspace/clear-ec-research
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2
PY=/workspace/venv/bin/python; S=experiments/lane_gpu1_20261106/source/frozen_features_20261106.py
$PY -u $S ssl5_f4 experiments/phase2_crossfold_20261015/fold4/ssl5/run/last.pt
$PY -u $S ext external
$PY -u $S ssl5_f0 experiments/phase2_crossfold_20261015/fold0/ssl5/run/last.pt
$PY -u $S ssl10_f4 experiments/phase2_backbone_ssl_20261020/fold4/ssl10/run/last.pt
$PY -u $S ssl20_f4 experiments/phase2_backbone_ssl_20261020/fold4/ssl20/run/last.pt
$PY -u $S tuned_f4 experiments/phase2_pooling_20261020/arms/std/runs/fold4/cpdino_vitb/epoch_20.pt
$PY -u $S tuned_f0 experiments/phase2_pooling_replication_20261026/f0_s46_std/runs/fold0/cpdino_vitb/epoch_20.pt
