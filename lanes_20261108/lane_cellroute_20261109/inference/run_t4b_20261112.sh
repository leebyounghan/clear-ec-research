#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-006): single-copy detector variants, replacing the kern / pkg runs of run_t4_20261112.sh (its streams 2 and 4 were
# stopped after their current job, so the main pod stays at 4 jobs x 1 CPU thread). Starts each when the job it replaces has exited.
#   fp16_b1  as fp16, the 13 non-pre-only detectors' single view as 1 copy instead of 4 identical copies
#   fp32_b1  as fp32, same
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109; G2=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; PY=/workspace/venv/bin/python
T="CLEAR_EC_TIER=t1 THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
run() {   # name, env, pid to wait for
  while kill -0 $3 2>/dev/null; do sleep 30; done
  RAMP=60 bash $G2 2 12000 "$T $2 timeout -k 10 14400 $PY $L/inference/test_t4.py 20 t4_$1_20 > $L/logs/t4_$1_20.log 2>&1 < /dev/null"
  echo "done $1 $(date -u)"
}
( run fp16_b1 "CLEAR_EC_PREC=fp16 CLEAR_EC_NO_TF32=1 CLEAR_EC_DET_COPIES=1" 593938 ) &
( run fp32_b1 "CLEAR_EC_PREC=fp32 CLEAR_EC_NO_TF32=1 CLEAR_EC_DET_COPIES=1" 594586 ) &
wait
echo "done t4b $(date -u)"
