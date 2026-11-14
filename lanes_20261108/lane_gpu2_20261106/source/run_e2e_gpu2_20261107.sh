#!/bin/bash
# Copy of lane GPU 1 run_e2e.sh for lane GPU 2: root moved to lane_gpu2_20261106/e2e/<arm>/f<fold>_s<seed>, and a check that the SSL arm finished. Steps: input, prepare, cpu tests, smoke, train, audit.
set -u
cd /workspace/clear-ec-research
L=experiments/lane_gpu1_20261106
ARM=$1; F=$2; S=${3:-46}; STOP=${4:-all}
case $F in
  0) SPLIT=experiments/ssl_screen_20260911/preflight/fold0.npz;;
  4) SPLIT=experiments/target_fold4_20260912/preflight/fold4.npz;;
  *) echo "fold $F not wired"; exit 2;;
esac
ROOT=experiments/lane_gpu2_20261106/e2e/$ARM/f${F}_s${S}; W=recipeA_ssl_arm_20261106.py; PY=/workspace/venv/bin/python
SSL=$L/ssl/$ARM/regression_input
A="--root $ROOT --split $SPLIT --fold $F --pooling std --seed $S --ssl-root $SSL --head-lr 0.001 --clip 1 --epochs 20 --schedule-epochs 10 --deadline-utc 2026-11-12T04:00:00Z"
RT=/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime
PP=$L/source:experiments/meta_ceiling_20261103/source:experiments/phase2_pooling_20261020/source:$RT
GPU=${CUDA_VISIBLE_DEVICES:-1}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_DEVICE_ORDER=PCI_BUS_ID
grep -qP "^$ARM\t0\t" $L/done.tsv || { echo "SSL arm $ARM has no rc 0 in done.tsv, refusing to start"; exit 3; }
mkdir -p $ROOT
step() { echo "### $(date -u +%FT%TZ) $1"; }
fail() { step "FAILED at $1 rc=$2"; echo "{\"arm\":\"$ARM\",\"fold\":$F,\"seed\":$S,\"status\":\"failed\",\"step\":\"$1\",\"rc\":$2}" > $ROOT/exit.json; exit $2; }
step input
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W make_input $ARM || fail input $?
step prepare
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W prepare $A || fail prepare $?
FZ=$ROOT/source/$W
step cpu-tests
CUDA_VISIBLE_DEVICES= PYTHONPATH=$RT $PY $FZ cpu-tests $A || fail cpu-tests $?
step smoke
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$RT $PY $FZ smoke $A || fail smoke $?
[ "$STOP" = smoke ] && { step "stopped after smoke"; exit 0; }
step train
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$RT $PY $FZ train $A || fail train $?
step audit
CUDA_VISIBLE_DEVICES= PYTHONPATH=$RT $PY $FZ audit $A || fail audit $?
step done
echo "{\"arm\":\"$ARM\",\"fold\":$F,\"seed\":$S,\"status\":\"completed\"}" > $ROOT/exit.json
