#!/bin/bash
# run_e2e.sh SSL_ARM FOLD [SEED] [ONLY_THROUGH]: recipe A end to end from this lane's SSL arm, fresh root e2e/<arm>/f<fold>_s<seed>. Steps: input, prepare, cpu tests, smoke, train, audit.
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
GPU=${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to this lane's GPU}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p $ROOT
step() { echo "### $(date -u +%FT%TZ) $1"; }
fail() { step "FAILED at $1 rc=$2"; echo "{\"arm\":\"$ARM\",\"fold\":$F,\"seed\":$S,\"status\":\"failed\",\"step\":\"$1\",\"rc\":$2}" > $ROOT/exit.json; exit $2; }
step input
# Keep regression_input only if it was built after this arm's SSL finished: GPU 2 built the fold 0 one from epoch 1 of 5 by mistake (2026-11-07).
[ -f $L/ssl/$ARM/run/complete.json ] || fail ssl-not-complete 3
if [ -f $SSL/run/complete.json ] && [ $SSL/run/complete.json -nt $L/ssl/$ARM/run/complete.json ]; then echo "regression_input exists and is newer than the SSL, not rewritten"; else CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W make_input $ARM || fail input $?; fi
step prepare
if [ -f $ROOT/manifest.json ]; then echo "manifest exists, not prepared again"; else CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP $PY $L/source/$W prepare $A || fail prepare $?; fi
FZ=$ROOT/source/$W
step cpu-tests
if [ -f $ROOT/cpu_tests.json ]; then echo "cpu tests already passed for this manifest"; else CUDA_VISIBLE_DEVICES= PYTHONPATH=$RT $PY $FZ cpu-tests $A || fail cpu-tests $?; fi
step smoke
if [ -f $ROOT/smoke.json ]; then echo "smoke already passed for this manifest"; else CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$RT $PY $FZ smoke $A || fail smoke $?; fi
[ "$STOP" = smoke ] && { step "stopped after smoke"; exit 0; }
step train
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$RT $PY $FZ train $A || fail train $?
step audit
CUDA_VISIBLE_DEVICES= PYTHONPATH=$RT $PY $FZ audit $A || fail audit $?
step done
echo "{\"arm\":\"$ARM\",\"fold\":$F,\"seed\":$S,\"status\":\"completed\"}" > $ROOT/exit.json
