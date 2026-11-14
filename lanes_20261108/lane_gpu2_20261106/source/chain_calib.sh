#!/bin/bash
# Calibration of the frozen screen, F1b remaining blocks, HEX-only exploratory read. Runs beside co-twin on GPU 2.
cd /workspace/clear-ec-research
L=experiments/lane_gpu2_20261106; PY=/workspace/venv/bin/python
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=2
$PY $L/source/corr_read_20261106.py legible_pred,legible_std,cls > $L/logs/corr_rest.log 2>&1; echo "$(date -u) corr rest rc $?"
$PY $L/source/corr_hexonly_20261106.py > $L/logs/corr_hexonly.log 2>&1; echo "$(date -u) hexonly rc $?"
$PY $L/source/calib_frozen_e2e_20261106.py > $L/logs/calib.log 2>&1; echo "$(date -u) calib rc $?"
echo "$(date -u) calib chain done"
