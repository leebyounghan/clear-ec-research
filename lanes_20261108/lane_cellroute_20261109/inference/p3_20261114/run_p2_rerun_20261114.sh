#!/bin/bash
# Same-time timing control for the p3 parity read (deploy lane, 2026-11-14): p2_all again (p2 context code read only, tier p2, the same settings as
# run_parity_p3_20261114.sh), output into package_20261114/p3/test/out/p2_all_rerun.json; then the p3 compare against this rerun.
cd /workspace/clear-ec-research || exit 1
H=experiments/lane_cellroute_20261109/inference/p3_20261114; T=experiments/lane_cellroute_20261109/package_20261114/p3/test
CODE=experiments/lane_cellroute_20261109/package_20261114/p2/context/code/experiments/lane_cellroute_20261109/inference
BASE="PYTHONPATH=/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime CLEAR_EC_PREC=fp16 CLEAR_EC_DET_COPIES=1 CLEAR_EC_NO_TF32=1 CLEAR_EC_UNLOAD=1 THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MALLOC_ARENA_MAX=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
ALL="CLEAR_EC_RO_DINOCV=1 CLEAR_EC_RO_RECIPEA=1 CLEAR_EC_RO_HEX6=1 CLEAR_EC_RO_CPSAM=1"
out=$T/out/p2_all_rerun.json; [ -e $out ] && { echo "exists $out"; exit 1; }
G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | grep -E '^ *(0|3),' | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
echo "$(date -u +%FT%TZ) start p2_all_rerun on GPU $G pid $$" >> $T/logs/queue.log
RAMP=30 timeout 5400 bash experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh $G 8000 \
  "env $BASE $ALL /workspace/venv/bin/python $H/p2_parity_20261114.py $CODE p2 $out 0,5,10,15,19" > $T/logs/p2_all_rerun.log 2>&1
echo "$(date -u +%FT%TZ) end p2_all_rerun rc $?" >> $T/logs/queue.log
/workspace/venv/bin/python $H/compare_p3_20261114.py $T/out/p3_all.json $out $T/compare_p3_vs_p2rerun.json >> $T/logs/queue.log 2>&1
/workspace/venv/bin/python $H/compare_p3_20261114.py $out experiments/lane_cellroute_20261109/package_20261114/p2/test/out/p2_all.json $T/compare_p2rerun_vs_p2.json >> $T/logs/queue.log 2>&1
echo "$(date -u +%FT%TZ) p2 rerun done" >> $T/logs/queue.log
