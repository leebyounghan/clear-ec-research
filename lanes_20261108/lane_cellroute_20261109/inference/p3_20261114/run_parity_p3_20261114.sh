#!/bin/bash
# Tier p3 / p3b parity run (deploy lane, 2026-11-14, D-COORD-P3-PREP): the same harness, frames and settings as the p2 parity run (p2_parity_20261114.py,
# a copy kept in this folder): 5 smoke frames idx 0,5,10,15,19, fp16b1 (CLEAR_EC_PREC=fp16 CLEAR_EC_DET_COPIES=1), TF32 off, 2 threads, unloading on,
# the four read-once switches on (as in the p2/p3 image), one process on GPU 0 or 3 (whichever has more free memory, 8000 MiB needed) through
# gpu_launch_20261111.sh. Reference = p2_all of the p2 run (package_20261114/p2/test/out/p2_all.json, same settings). usage: run_parity_p3_20261114.sh VARIANT
V=${1:?variant}; cd /workspace/clear-ec-research || exit 1
H=experiments/lane_cellroute_20261109/inference/p3_20261114; T=experiments/lane_cellroute_20261109/package_20261114/$V/test
CODE=experiments/lane_cellroute_20261109/package_20261114/$V/context/code/experiments/lane_cellroute_20261109/inference
IDX=${IDX:-0,5,10,15,19}; mkdir -p $T/out $T/logs; out=$T/out/${V}_all.json
test -f $CODE/manifests/deploy_$V.json || { echo "no $V context"; exit 1; }
BASE="PYTHONPATH=/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime CLEAR_EC_PREC=fp16 CLEAR_EC_DET_COPIES=1 CLEAR_EC_NO_TF32=1 CLEAR_EC_UNLOAD=1 THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MALLOC_ARENA_MAX=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
ALL="CLEAR_EC_RO_DINOCV=1 CLEAR_EC_RO_RECIPEA=1 CLEAR_EC_RO_HEX6=1 CLEAR_EC_RO_CPSAM=1"
if [ -f $out ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$out'))['done'] else 1)" 2>/dev/null; then echo "$out done already"; exit 0; fi
G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | grep -E '^ *(0|3),' | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
echo "$(date -u +%FT%TZ) start ${V}_all on GPU $G pid $$" >> $T/logs/queue.log
RAMP=30 timeout 5400 bash experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh $G 8000 \
  "env $BASE $ALL /workspace/venv/bin/python $H/p2_parity_20261114.py $CODE $V $out $IDX" > $T/logs/${V}_all.log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) end ${V}_all rc $rc" >> $T/logs/queue.log
/workspace/venv/bin/python $H/compare_p3_20261114.py $out experiments/lane_cellroute_20261109/package_20261114/p2/test/out/p2_all.json $T/compare_${V}_vs_p2.json >> $T/logs/queue.log 2>&1
echo "$(date -u +%FT%TZ) ${V} parity done" >> $T/logs/queue.log
