#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-004): emu smoke after the attention fix + FLOP count per stage (bf16), 1 frame each, GPU 2, 2 CPU threads each
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109; G2=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; PY=/workspace/venv/bin/python
RAMP=20 bash $G2 2 20000 "CLEAR_EC_TIER=t1 CLEAR_EC_PREC=emu CLEAR_EC_NO_TF32=1 THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout -k 10 2400 $PY $L/inference/test_t4.py 1 t4_smoke_emu2 > $L/logs/t4_smoke_emu2.log 2>&1 < /dev/null" &
sleep 30
RAMP=20 bash $G2 2 12000 "CLEAR_EC_TIER=t1 CLEAR_EC_PREC=bf16 FLOPS=1 THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout -k 10 2400 $PY $L/inference/test_t4.py 1 t4_flops_bf16 > $L/logs/t4_flops_bf16.log 2>&1 < /dev/null" &
wait
echo "done smoke2 $(date -u)"
