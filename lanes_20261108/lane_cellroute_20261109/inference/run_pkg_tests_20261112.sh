#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-001): (1) 3-frame packaged census rerun after the digest patch (GPU 3 guard), then (2) 20 Grand Challenge style jobs
# through docker/gc_entry.py on the idle GPU 2 (general guard), one job at a time, <= 4 CPU threads
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109
bash experiments/lane_backbone_20261108/source/gpu3_launch_20261110.sh 25000 "CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 3600 /workspace/venv/bin/python $L/inference/test_packaged.py 3 packaged_parity_3b > $L/logs/packaged_parity_3b.log 2>&1 < /dev/null"
grep -q '^SUMMARY' $L/logs/packaged_parity_3b.log || { echo "stop: census rerun had no SUMMARY $(date -u)"; exit 1; }
echo "done census $(date -u)"
bash experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh 2 25000 "timeout -k 10 10800 /workspace/venv/bin/python $L/inference/docker/test_gc_jobs.py 20 gc_jobs_20 > $L/logs/gc_jobs_20.log 2>&1 < /dev/null"
echo "done gc jobs rc $? $(date -u)"
