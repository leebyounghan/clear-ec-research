#!/bin/bash
# E3 fold 1 (SSL root differs from fold 0): box-restricted std pooling recipe A, GPUs 0 and 1 via torchrun (lane01 job).
cd /workspace/clear-ec-research
F=1; PY=/workspace/venv/bin/python; ROOT=experiments/green_day3_20261103/e3_boxpool/f${F}_s46_stdbox; W=$ROOT/source/recipeA_box_pool_backbone_20261103.py
SPLIT=experiments/ssl_screen_20260911/preflight/fold${F}.npz; SSL=experiments/oof_recipeA_20261027/fold${F}/ssl5
ARGS="--root $ROOT --split $SPLIT --fold $F --pooling std_box --seed 46 --ssl-epochs 5 --ssl-root $SSL --head-lr 0.001 --clip 1 --epochs 20 --schedule-epochs 10 --deadline-utc none"
mkdir -p $ROOT/source; cp experiments/green_day3_20261103/e3_boxpool/recipeA_box_pool_backbone_20261103.py $W
LOG=$ROOT/driver.log
{ echo "### prepare $(date -u)"; CUDA_VISIBLE_DEVICES="" $PY $W prepare $ARGS; } >> $LOG 2>&1 || { echo "FAILED prepare" >> $LOG; exit 1; }
{ echo "### cpu_tests $(date -u)"; CUDA_VISIBLE_DEVICES="" $PY $W cpu_tests $ARGS; } >> $LOG 2>&1 || { echo "FAILED cpu_tests" >> $LOG; exit 1; }
{ echo "### smoke $(date -u)"; $PY -m torch.distributed.run --standalone --nproc_per_node=2 $W smoke $ARGS; } >> $LOG 2>&1 || { echo "FAILED smoke" >> $LOG; exit 1; }
{ echo "### train $(date -u)"; $PY -m torch.distributed.run --standalone --nproc_per_node=2 $W train $ARGS; } >> $LOG 2>&1 || { echo "FAILED train" >> $LOG; exit 1; }
{ echo "### audit $(date -u)"; CUDA_VISIBLE_DEVICES="" $PY $W audit $ARGS; } >> $LOG 2>&1 || { echo "FAILED audit" >> $LOG; exit 1; }
echo "e3 fold $F done $(date -u)" >> $LOG
