#!/bin/bash
# deploy lane 2026-11-12 (D-GPU-001): score impact of the T4 default on 200 labelled rows (inference/rows200_20261112.npy), tier t1.
# GPUs 0, 1, 3 (free since 22:32 EST), one stream each, 1 CPU thread per job (lane cap 4). Each stream: its third of the rows, reference then T4 path.
#   ref    original fp32 weight files, bf16 (the lanes' numerics)
#   t4d    packaged, fp16 + one detector copy, TF32 off (the container default below compute capability 8.0)
# CLEAR_EC_UNLOAD=0 keeps built models across frames (same numerics, about half the time).
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109; G=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; PY=/workspace/venv/bin/python
T="CLEAR_EC_TIER=${TIERX:-t1} CLEAR_EC_UNLOAD=0 ROWS_FILE=$L/inference/rows200_20261112.npy THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
TAG=${TIERX:-t1}
run() {   # gpu, name, env, first, n
  RAMP=120 bash $G $1 30000 "$T $3 timeout -k 10 21600 $PY $L/inference/test_t4.py $5 t4_${2}200_${TAG}_c$4 $4 > $L/logs/t4_${2}200_${TAG}_c$4.log 2>&1 < /dev/null"
  echo "done $2 c$4 gpu$1 $(date -u)"
}
( run 0 ref "REF32=1 CLEAR_EC_PREC=bf16" 0 67; run 0 t4d "CLEAR_EC_PREC=fp16 CLEAR_EC_DET_COPIES=1 CLEAR_EC_NO_TF32=1" 0 67 ) &
( run 1 ref "REF32=1 CLEAR_EC_PREC=bf16" 67 67; run 1 t4d "CLEAR_EC_PREC=fp16 CLEAR_EC_DET_COPIES=1 CLEAR_EC_NO_TF32=1" 67 67 ) &
( run 3 ref "REF32=1 CLEAR_EC_PREC=bf16" 134 66; run 3 t4d "CLEAR_EC_PREC=fp16 CLEAR_EC_DET_COPIES=1 CLEAR_EC_NO_TF32=1" 134 66 ) &
wait
echo "done t4 200 $TAG $(date -u)"
