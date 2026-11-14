#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-002): per-tier runs on GPU 2 after run_pkg_tests2 has finished (one job at a time, <= 4 CPU threads on the main pod):
# (1) 3-frame packaged runs (census, GPU peak, shared inputs) for full, t1, t2, t3 with unloading; (2) 5 Grand Challenge style jobs per tier
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109; G2=experiments/lane_backbone_20261108/source/gpu_launch_20261111.sh
for i in $(seq 1 120); do grep -q 'done fp16 path' $L/logs/run_pkg_tests2_20261112.log && break; sleep 60; done
echo "start tiers $(date -u)"
for t in full t1 t2 t3; do
  TE=""; [ "$t" != "full" ] && TE="CLEAR_EC_TIER=$t"
  bash $G2 2 25000 "env $TE CLEAR_EC_UNLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 timeout -k 10 3600 /workspace/venv/bin/python $L/inference/test_packaged.py 3 packaged_tier_${t}_3 > $L/logs/packaged_tier_${t}_3.log 2>&1 < /dev/null"
  echo "done packaged $t $(date -u)"
done
for t in full t1 t2 t3; do
  TE=""; [ "$t" != "full" ] && TE="CLEAR_EC_TIER=$t"
  bash $G2 2 25000 "env $TE CLEAR_EC_UNLOAD=1 timeout -k 10 3600 /workspace/venv/bin/python $L/inference/docker/test_gc_jobs.py 5 gc_jobs_5_${t} > $L/logs/gc_jobs_5_${t}.log 2>&1 < /dev/null"
  echo "done gc $t $(date -u)"
done
echo "done tiers $(date -u)"
