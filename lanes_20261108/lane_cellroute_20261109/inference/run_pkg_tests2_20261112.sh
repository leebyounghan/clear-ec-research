#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-001/002), one job at a time (<= 4 CPU threads on the main pod):
# (1) 3-frame packaged census rerun (GPU 3 guard); (2) 20 Grand Challenge style jobs with model unloading on the idle GPU 2;
# (3) T4 path: 20 frames, fp16 autocast forced + unloading, vs fp32 and vs the bf16 packaged run (GPU 2)
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109; G2=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh
bash experiments/lane_backbone_20261108/source/gpu3_launch_20261110.sh 25000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 3600 /workspace/venv/bin/python $L/inference/test_packaged.py 3 packaged_parity_3b > $L/logs/packaged_parity_3b.log 2>&1 < /dev/null"
grep -q '^SUMMARY' $L/logs/packaged_parity_3b.log && echo "done census $(date -u)" || echo "census rerun had no SUMMARY $(date -u) (continuing)"
bash $G2 2 25000 "CLEAR_EC_UNLOAD=1 timeout -k 10 10800 /workspace/venv/bin/python $L/inference/docker/test_gc_jobs.py 20 gc_jobs_20_unload > $L/logs/gc_jobs_20_unload.log 2>&1 < /dev/null"
echo "done gc jobs rc $? $(date -u)"
bash $G2 2 25000 "CLEAR_EC_UNLOAD=1 CLEAR_EC_AUTOCAST=fp16 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_packaged.py 20 packaged_fp16_unload_20 > $L/logs/packaged_fp16_unload_20.log 2>&1 < /dev/null"
echo "done fp16 path rc $? $(date -u)"
