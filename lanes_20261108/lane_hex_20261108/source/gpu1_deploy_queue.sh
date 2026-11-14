#!/bin/bash
# HEX lane GPU 1 deploy builds (2026-11-10, coordinator's rehearsal list). Waits for pid arg 1 (CPSAM fold-2 rerun) to exit, then
# runs one job at a time on GPU 1: the 4 missing fall16 detectors, the full-data graph networks (pre47_s10, pre48_s10, HEX-only
# pre46 x5), and the full-data CV-weighted DINOv3 (cvw3, epochs 1-12). STOP_DEPLOY_QUEUE skips the remaining jobs.
# Log: logs/gpu1_deploy_queue.log. Do not edit while it runs.
L=/workspace/clear-ec-research/experiments/lane_hex_20261108; Q=$L/logs/gpu1_deploy_queue.log; cd /workspace/clear-ec-research; PY=/workspace/venv/bin/python
P=${1:-0}; echo "$(date -u +%FT%TZ) waiting for pid $P" >> $Q; while [ "$P" != 0 ] && [ -d /proc/$P ]; do sleep 30; done
run() { n=$1; shift; [ -f $L/STOP_DEPLOY_QUEUE ] && { echo "$(date -u +%FT%TZ) STOP_DEPLOY_QUEUE: skip $n" >> $Q; return; }
  echo "$(date -u +%FT%TZ) start $n" >> $Q; t0=$(date +%s); env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 "$@" > $L/logs/deploy_$n.log 2>&1; rc=$?
  echo "$(date -u +%FT%TZ) end $n rc $rc seconds $(( $(date +%s) - t0 ))" >> $Q; }
run det_sg2_specialist_s51 $PY $L/source/deploy_det.py sg2_specialist_s51 --seed 51
run det_sg2_specialist_s52 $PY $L/source/deploy_det.py sg2_specialist_s52 --seed 52
run det_sg2_sigma3p5 $PY $L/source/deploy_det.py sg2_sigma3p5 --seed 46 --sigma 3.5
run det_sg2_sigma5 $PY $L/source/deploy_det.py sg2_sigma5 --seed 46 --sigma 5
run graph_pre47_s10 $PY $L/source/graph_train_full.py full_d64_l5_pre47_s10 --graphs cell_graphs_pre47.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 10 --no-ctx --wd 0.01 --drop 0.1 --folds full
run graph_pre48_s10 $PY $L/source/graph_train_full.py full_d64_l5_pre48_s10 --graphs cell_graphs_pre48.npz --d 64 --layers 5 --epochs 25 --lr 1e-3 --seeds 10 --no-ctx --wd 0.01 --drop 0.1 --folds full
run graph_hex_pre46_s5 $PY $L/source/gnn_hex.py full_hex_pre46_s5 --graphs /workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/cell_graphs.npz --lw 0,0,1 --seeds 5 --threads 2 --folds full
run cvw3_f99_s46 env CD_LOSS=cvw3 $PY $L/source/family_cvw3_copy.py all --family dinov3_b16 --fold 99 --seed 46 --epochs 12 --schedule-epochs 10 --micro 16 --workers 4 --ssl-from experiments/lane_cd_20261108/arms/dinov3_b16_f99_s46/ssl/last.pt --root experiments/lane_hex_20261108/deploy/cvw3_f99_s46
echo "$(date -u +%FT%TZ) queue done" >> $Q
