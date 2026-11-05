#!/bin/bash
# Bridge-capacity member: recipe A std s46 with BRIDGE_VARIANT (enc_lr5x | token_mlp) on one fold (two GPUs via torchrun).
# usage: run_bridge.sh FOLD VARIANT   (root experiments/meta_ceiling_20261103/bridge/f<FOLD>_<VARIANT>)
cd /workspace/clear-ec-research
F=$1; V=$2; PY=/workspace/venv/bin/python
SRC=experiments/meta_ceiling_20261103/source; W=$SRC/recipeA_bridge_capacity_20261103.py
ROOT=experiments/meta_ceiling_20261103/bridge/f${F}_${V}
if [ "$F" = 4 ]; then SPLIT=experiments/target_fold4_20260912/preflight/fold4.npz; else SPLIT=experiments/ssl_screen_20260911/preflight/fold${F}.npz; fi
SSL=experiments/phase2_crossfold_20261015/fold${F}/ssl5
[ "$F" = 1 ] || [ "$F" = 2 ] || [ "$F" = 3 ] && SSL=experiments/oof_recipeA_20261027/fold${F}/ssl5
ARGS="--root $ROOT --split $SPLIT --fold $F --pooling std --seed 46 --ssl-root $SSL --head-lr 0.001 --clip 1 --epochs 20 --schedule-epochs 10 --deadline-utc 2026-11-06T12:00:00Z"
export BRIDGE_VARIANT=$V
mkdir -p $ROOT; LOG=$ROOT/driver.log
step() { local name=$1; shift; echo "### $name $(date -u)" >> $LOG; "$@" >> $LOG 2>&1 || { echo "FAILED $name" >> $LOG; exit 1; }; }
[ -f $ROOT/manifest.json ] || step prepare env CUDA_VISIBLE_DEVICES="" $PY $W prepare $ARGS
[ -f $ROOT/cpu_tests.json ] || step cpu_tests env CUDA_VISIBLE_DEVICES="" $PY $W cpu_tests $ARGS
[ -f $ROOT/smoke.json ] || step smoke $PY -m torch.distributed.run --standalone --nproc_per_node=2 $W smoke $ARGS
[ -f $ROOT/runs/fold$F/cpdino_vitb/complete.json ] || step train $PY -m torch.distributed.run --standalone --nproc_per_node=2 $W train $ARGS
[ -f $ROOT/audit.json ] || step audit env CUDA_VISIBLE_DEVICES="" $PY $W audit $ARGS
echo "bridge fold $F $V done $(date -u)" >> $LOG
