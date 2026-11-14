#!/bin/bash
# run_arm.sh ARM FOLD SEED EPOCHS SCHEDULE WRAPPER: prepare, cpu tests, smoke, train, audit in a fresh root. ARM_CONFIG and other wrapper settings come from the environment.
set -u
cd /workspace/clear-ec-research
L=experiments/lane_cv_20261108
ARM=$1; F=$2; S=$3; EP=$4; SC=$5; W=$6
ROOT=$L/arms/$ARM/f${F}_s${S}
case $F in
  0) SPLIT=experiments/ssl_screen_20260911/preflight/fold0.npz;;
  4) SPLIT=experiments/target_fold4_20260912/preflight/fold4.npz;;
  *) echo "fold $F not wired"; exit 2;;
esac
SSL=experiments/phase2_crossfold_20261015/fold$F/ssl5
PY=/workspace/venv/bin/python
A="--root $ROOT --split $SPLIT --fold $F --pooling std --seed $S --ssl-root $SSL --head-lr 0.001 --clip 1 --epochs $EP --schedule-epochs $SC --deadline-utc 2026-11-12T04:00:00Z"
PP=$L/source:experiments/lane_gpu0_20261106/source:experiments/meta_ceiling_20261103/source:experiments/phase2_pooling_20261020/source:${PYTHONPATH:-}
mkdir -p $ROOT $L/logs
LOG=$L/logs/${ARM}_f${F}_s${S}.log
step() { echo "### $(date -u +%FT%TZ) $1" >> $LOG; }
fail() { step "FAILED at $1 rc=$2"; echo "{\"arm\":\"$ARM\",\"fold\":$F,\"seed\":$S,\"status\":\"failed\",\"step\":\"$1\",\"rc\":$2}" > $ROOT/exit.json; exit $2; }
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

step prepare
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W prepare $A >> $LOG 2>&1 || fail prepare $?
FZ=$ROOT/source/$W
step cpu-tests
CUDA_VISIBLE_DEVICES= $PY $FZ cpu-tests $A >> $LOG 2>&1 || fail cpu-tests $?
step no-keepalive-stop
true
step smoke
CUDA_VISIBLE_DEVICES=2 $PY $FZ smoke $A >> $LOG 2>&1 || fail smoke $?
step train
CUDA_VISIBLE_DEVICES=2 $PY $FZ train $A >> $LOG 2>&1 || fail train $?
step audit
CUDA_VISIBLE_DEVICES= $PY $FZ audit $A >> $LOG 2>&1 || fail audit $?
step done
echo "{\"arm\":\"$ARM\",\"fold\":$F,\"seed\":$S,\"status\":\"completed\"}" > $ROOT/exit.json
