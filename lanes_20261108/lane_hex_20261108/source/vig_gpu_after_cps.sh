#!/bin/bash
# Start the Vigueras full run on GPU 1 only after the full-data CPSAM build (cps_full.sh, pid 370257) has exited, so it never
# shares the GPU with that deploy-critical job (HEX lane, 2026-11-10). usage: bash vig_gpu_after_cps.sh
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; G=$L/logs/vig_gpu.log
echo "$(date -u +%FT%TZ) waiting for cps_full.sh pid 370257" >> $G
while kill -0 370257 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) cps_full ended; start vig_infer" >> $G
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 /workspace/venv/bin/python -I $L/source/vig_infer.py $L/runs/vig_maps >> $G 2>&1
echo "$(date -u +%FT%TZ) vig_infer rc $?" >> $G
