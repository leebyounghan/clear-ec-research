#!/bin/bash
cd /workspace/clear-ec-research; L=experiments/lane_gpu0_20261106
ROOT=$L/arms/rope_dryrun/f4_s46; W=recipeA_rope_fixed_20261106.py; PY=/workspace/venv/bin/python
A="--root $ROOT --split experiments/target_fold4_20260912/preflight/fold4.npz --fold 4 --pooling std --seed 46 --ssl-root experiments/phase2_crossfold_20261015/fold4/ssl5 --head-lr 0.001 --clip 1 --epochs 20 --schedule-epochs 10 --deadline-utc 2026-11-12T04:00:00Z"
PP=$L/source:experiments/meta_ceiling_20261103/source:experiments/phase2_pooling_20261020/source:${PYTHONPATH:-}
export OMP_NUM_THREADS=2
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W prepare $A > $L/logs/rope_dry.log 2>&1 || { echo prepare failed; tail -5 $L/logs/rope_dry.log; exit 1; }
CUDA_VISIBLE_DEVICES= $PY $ROOT/source/$W cpu-tests $A >> $L/logs/rope_dry.log 2>&1 || { echo cpu-tests failed; tail -5 $L/logs/rope_dry.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY $ROOT/source/$W smoke $A >> $L/logs/rope_dry.log 2>&1 || { echo smoke failed; tail -5 $L/logs/rope_dry.log; exit 1; }
/workspace/venv/bin/python -c "import json;d=json.load(open('$ROOT/smoke.json'));print('smoke',d['status'],d['single_gpu_reference_gradient_check']);m=json.load(open('$ROOT/manifest.json'));print(m['recipe']['rope_rescale_train'],m['execution']['world_size'])"
rm -r $L/arms/rope_dryrun
echo DRYRUN_OK
