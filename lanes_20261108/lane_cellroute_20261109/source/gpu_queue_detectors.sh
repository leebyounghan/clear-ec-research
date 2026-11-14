#!/bin/bash
# Cell-route lane, 2026-11-09: full-data pre-only sg2 detectors s47, s48 on GPU 1 (HEX lane OK), one job at a time, then the 200-frame check.
cd /workspace/clear-ec-research
L=experiments/lane_cellroute_20261109; mkdir -p $L/deploy/detectors $L/logs
ENC=experiments/fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt
for SEED in 47 48; do
  CUDA_VISIBLE_DEVICES=1 /workspace/venv/bin/python experiments/green_track_20261028/source/green_external_srep_20261030.py pretrain --kind sg2 --seed $SEED --updates 2000 --scale 1.296 --encoder $ENC --deploy-out $L/deploy/detectors/sg2pre_s$SEED.pt > $L/logs/pretrain_s$SEED.log 2>&1 || { echo "pretrain s$SEED failed" >> $L/logs/gpu_queue.log; exit 1; }
  echo "pretrain s$SEED done $(date)" >> $L/logs/gpu_queue.log
done
CUDA_VISIBLE_DEVICES=1 /workspace/venv/bin/python $L/source/check200.py > $L/logs/check200.log 2>&1 && echo "check200 done $(date)" >> $L/logs/gpu_queue.log || echo "check200 failed" >> $L/logs/gpu_queue.log
