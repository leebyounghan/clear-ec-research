#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-003): primary tier p first, then the t2 / t3 census; GPU 2, one job at a time (<= 4 CPU threads on the main pod)
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109; G2=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh
while kill -0 585476 2>/dev/null; do sleep 20; done          # the t1 3-frame census of the stopped chain
echo "start p $(date -u)"
bash $G2 2 25000 "CLEAR_EC_TIER=p CLEAR_EC_UNLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/test_packaged.py 20 packaged_tier_p_20 > $L/logs/packaged_tier_p_20.log 2>&1 < /dev/null"
echo "done p in-process $(date -u)"
bash $G2 2 25000 "CLEAR_EC_TIER=p CLEAR_EC_UNLOAD=1 timeout -k 10 7200 /workspace/venv/bin/python $L/inference/docker/test_gc_jobs.py 20 gc_jobs_20_p > $L/logs/gc_jobs_20_p.log 2>&1 < /dev/null"
echo "done p jobs $(date -u)"
for t in t2 t3; do
  bash $G2 2 25000 "CLEAR_EC_TIER=$t CLEAR_EC_UNLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 3600 /workspace/venv/bin/python $L/inference/test_packaged.py 3 packaged_tier_${t}_3 > $L/logs/packaged_tier_${t}_3.log 2>&1 < /dev/null"
  echo "done census $t $(date -u)"
done
echo "done tier chain $(date -u)"
