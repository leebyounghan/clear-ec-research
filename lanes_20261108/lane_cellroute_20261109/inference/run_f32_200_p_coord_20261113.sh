#!/bin/bash
# coordinator 2026-11-13 (PI request: "look at fp32 once more"): tier p, the same 200 labelled rows as run_t4_200_20261112.sh, two more precision paths
# against the existing reference out/t4_ref200_p*.json. GPUs 1, 2, 3 (keepalive yields), one stream each, 1 CPU thread per job.
#   f32d  every autocast region fp32, CPSAM float32, TF32 off, one detector copy
#   mixd  fp16, detector / hex6 modules fp32, CPSAM float32, TF32 off, one detector copy
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109; G=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh; PY=/workspace/venv/bin/python
T="CLEAR_EC_TIER=p CLEAR_EC_UNLOAD=0 ROWS_FILE=$L/inference/rows200_20261112.npy THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
MODS=deploy_hexkonan_image,hex6_correction,green_point_features_20261027,green_sg2_specialist_20261028,green_s8_specialist_seed_20261028,green_selftrain_20261028
run() {   # gpu, name, env, first, n
  RAMP=120 bash $G $1 30000 "$T $3 timeout -k 10 21600 $PY $L/inference/test_t4.py $5 t4_${2}200_p_c$4 $4 > $L/logs/t4_${2}200_p_c$4.log 2>&1 < /dev/null"
  echo "done $2 c$4 gpu$1 $(date -u)"
}
F="CLEAR_EC_PREC=fp32 CLEAR_EC_NO_TF32=1 CLEAR_EC_DET_COPIES=1"
M="CLEAR_EC_PREC=mix CLEAR_EC_FP32_MODULES=$MODS CLEAR_EC_NO_TF32=1 CLEAR_EC_DET_COPIES=1"
( run 1 f32d "$F" 0 67;   run 1 mixd "$M" 0 67 ) &
( run 2 f32d "$F" 67 67;  run 2 mixd "$M" 67 67 ) &
( run 3 f32d "$F" 134 66; run 3 mixd "$M" 134 66 ) &
wait
echo "done f32 200 p $(date -u)"
